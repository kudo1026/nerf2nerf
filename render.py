import argparse
import os

import torch
import yaml
import numpy as np

import utils as uu
from nerf_wrapper import NeRFWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', default='max_planck')
    parser.add_argument('-ab', '--a_or_b', default='a')
    parser.add_argument('--i', type=int, default=0)
    args = parser.parse_args()
    with open(os.path.join(os.getcwd(), 'options', args.yaml + ".yaml"), "r") as f:
        opt = uu.AttributeDict(yaml.load(f, Loader=yaml.FullLoader))
    root = os.path.join(os.getcwd(), "scenes", "scene_" + str(opt.scene_no))
    with torch.cuda.device(opt.device):
        wrapper = NeRFWrapper(os.path.join(root, args.a_or_b, f'nerf_model_{args.a_or_b}.pt'), os.path.join(root, args.a_or_b, f'distilled_{args.a_or_b}.ckpt'), 'jit')
        # camera_angle_x in json is fov_x
        # poses are stored in json as 4x4 c2w of spec x->right, y->up, z->back (so called "raw pose")
        # loaded poses are 3x4 c2w of spec x->right, y->down, z->front
        focal, _ = uu.load_poses(os.path.join(root, 'transforms.json'), opt.image_size)
        poses = np.array(uu.gen_hemispherical_poses(1, np.pi / 6, gamma_hi=np.pi / 3, target=np.array([0, 0, 0.5]), m=3, n=16, pose_spec=1))
        poses = torch.as_tensor(poses, device=opt.device)
        output_dir = os.path.join(root, args.a_or_b)
        os.makedirs(output_dir, exist_ok=True)
        if not args.i:
            meta = {f'{i:03d}': poses[i] for i in range(48)}
            torch.save(meta, os.path.join(root, f'{args.a_or_b}_in.pt'))
        wrapper.render_image_fine(poses[args.i], focal_length=focal, save_p=output_dir, index=args.i, near=opt.near, far=opt.far, chunk=opt.chunk)
