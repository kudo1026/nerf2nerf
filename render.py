import argparse
import json
import os

import numpy as np
import torch
import yaml

import utils as uu
from my_utils.utils import complete_transformation
from nerf_wrapper import NeRFWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', default='max_planck')
    parser.add_argument('--layout', default='a')
    parser.add_argument('--rel_trans', default='pred', choices=['gt', 'pred'])
    parser.add_argument('--i', type=int, default=0)
    args = parser.parse_args()
    with open(os.path.join('options', args.obj + '.yaml')) as f:
        opt = uu.AttributeDict(yaml.load(f, Loader=yaml.FullLoader))
    root = os.path.join('scenes', 'scene_' + str(opt.scene_no))
    output_dir = os.path.join(root, args.layout)
    os.makedirs(output_dir, exist_ok=True)
    with torch.cuda.device(opt.device):
        wrapper = NeRFWrapper(os.path.join(root, args.layout, f'nerf_model_{args.layout}.pt'), os.path.join(root, args.layout, f'distilled_{args.layout}.ckpt'), 'jit')
        # camera_angle_x in json is fov_x
        # poses are stored in json as 4x4 c2w of spec x->right, y->up, z->back (so called 'raw pose')
        # loaded poses are 3x4 c2w of spec x->right, y->down, z->front
        focal, poses = uu.load_poses(os.path.join(root, 'transforms.json'), opt.image_size)
        n = len(poses)
        # poses = np.array(uu.gen_hemispherical_poses(1, np.pi / 6, gamma_hi=np.pi / 4, target=np.array([0, 0, 0.4]), m=3, n=16, pose_spec=1))
        # poses = torch.as_tensor(poses, device=opt.device)
        assert 0 <= args.i < n
        if not args.i:
            meta = {f'{i:03d}': poses[i] for i in range(len(poses))}
            torch.save(meta, os.path.join(root, f'{args.layout}_in.pt'))
        pose = poses[args.i]
        if args.layout == 'b':
            assert args.rel_trans
            if args.rel_trans == 'gt':
                with open(os.path.join(root, 'gt_transform.json')) as file:
                    gt_poses = json.load(file)
                # model to A
                A = np.array(gt_poses[args.obj]['scene_a'])
                # model to B
                B = np.array(gt_poses[args.obj]['scene_b'])
                T_AB = torch.as_tensor(B @ np.linalg.inv(A), dtype=torch.float32)
            elif args.rel_trans == 'pred':
                T_AB = np.load(os.path.join(root, 'output/output_transform.npy'))
            pose = T_AB @ complete_transformation(pose)
        wrapper.render_image_fine(pose, focal_length=focal, save_p=output_dir, index=args.i, near=opt.near, far=opt.far, chunk=opt.chunk)
