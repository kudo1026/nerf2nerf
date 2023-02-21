from dataclasses import dataclass, field
from typing import Type, Union

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from torchtyping import TensorType
from typing_extensions import Literal
import math


@dataclass
class MyNerfactoModelConfig(NerfactoModelConfig):
    """My Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: MyNerfactoModel)


class MyNerfactoModel(NerfactoModel):
    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)[0]
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = field_outputs[FieldHeadNames.RGB]
        img = self.renderer_rgb(rgb=rgb, weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        rgb = torch.cat((torch.empty_like(rgb[..., [0], :]), rgb), dim=-2)
        weights = torch.cat((torch.zeros_like(weights[..., [0], :]), weights), dim=-2)
        deltas = torch.cat((ray_samples.frustums.starts[..., [0], :], ray_samples.deltas), dim=-2)

        outputs = {
            'rgb': rgb,
            'rgb_img': img,
            'accumulation': accumulation,
            'depth': depth,
            'weights': weights,
            'deltas': deltas,
            'directions': ray_samples.frustums.directions[:, 0]
        }
        return outputs

    def call_eval(self, pts, *args):
        """ pts: tensor of shape N*3 """
        return None, self.field.density_fn(pts)

    def transmittance_eval(self, pts, cams, chunk_size=30000):
        """ pts: tensor of shape N*3,
            cams: tensor of shape N*3 """
        n = len(pts)
        m = math.ceil(n / chunk_size)
        ts_lst = []
        for i in range(m):
            if i < m - 1:
                cur_pts = pts[i * chunk_size:(i + 1) * chunk_size]
                cur_cams = cams[i * chunk_size:(i + 1) * chunk_size]
            else:
                cur_pts = pts[i * chunk_size:]
                cur_cams = cams[i * chunk_size:]
            cur_n = len(cur_pts)
            cur_pts -= cur_cams
            dists = torch.linalg.norm(cur_pts, dim=1, keepdim=True)
            dirs = cur_pts / dists
            ray_bundle = RayBundle(
                origins=cur_cams,
                directions=dirs,
                pixel_area=torch.empty_like(cur_pts[..., [0]]),
                camera_indices=torch.empty_like(cur_pts[..., [0]]),
                nears=torch.full_like(cur_pts[..., [0]], 0.05),
                fars=torch.full_like(cur_pts[..., [0]], 10.0),
            )
            ray_samples = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)[0]
            field_outputs = self.field(ray_samples)
            xs = torch.cat((ray_samples.frustums.starts[..., 0], ray_samples.frustums.ends[:, [-1], 0]), dim=1)  # n*49
            indices = torch.searchsorted(xs, dists)  # n*1 in [0, 49]
            zeros = torch.zeros_like(cur_pts[..., [0]], dtype=torch.float32)
            cs = torch.cumsum(field_outputs[FieldHeadNames.DENSITY] * ray_samples.deltas, dim=1)[..., 0]  # n*48
            cs = torch.cat((zeros.expand(-1, 2), cs), dim=1)    # n*50
            sigmas = field_outputs[FieldHeadNames.DENSITY][..., 0]
            sigmas = torch.cat((zeros, sigmas, zeros), dim=1)   # n*50
            xs = torch.cat((zeros, xs), dim=1)  # n*50
            rs = torch.arange(cur_n)[:, None]
            ts = torch.exp(-(cs[rs, indices] + sigmas[rs, indices] * (dists - xs[rs, indices])))  # n*1
            ts_lst.append(ts[:, 0])
        return torch.cat(ts_lst)


class MyRGBRenderer(RGBRenderer):
    """Weighted volumetic rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    @classmethod
    def combine_rgb(
        cls,
        rgb: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        ws: TensorType["bs":..., "num_samples", 1],
        bg_w,
        background_color: Union[Literal["random", "last_sample"], TensorType[3]] = "random"
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            weights: Termination probability mass for each sample.
            ws: Weights for each sample. E.g. from IDW.
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        comp_rgb = torch.sum(rgb * weights * ws, dim=-2)
        accumulation = torch.sum(weights, dim=-2)

        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        elif background_color == "random":
            background_color = torch.rand_like(comp_rgb)

        comp_rgb += background_color * (1 - accumulation) * bg_w

        return comp_rgb

    def forward(
        self,
        rgb: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        ws: TensorType["bs":..., "num_samples", 1],
        bg_w
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            weights: Termination probability mass for each sample.
            ws: Weights for each sample. E.g. from IDW.

        Returns:
            Outputs of rgb values.
        """

        rgb = self.combine_rgb(rgb, weights, ws, bg_w, background_color=self.background_color)
        return rgb
