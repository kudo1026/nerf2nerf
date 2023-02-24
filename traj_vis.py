import json

import numpy as np
import torch

from my_utils.visualizer import Visualizer

scene_no = 4
layout = 'b'

transforms_path = f'/home/slin/projects/nerf2nerf/scenes/scene_{scene_no}/transforms.json'
with open(transforms_path) as f:
    transforms = json.load(f)
poses = [np.array(frame['transform_matrix']) for frame in transforms['frames']]

annotation_poses = []
for i in range(1, 3):
    transforms_path = f'/home/slin/projects/nerf2nerf/scenes/scene_{scene_no}/{layout}/{i}.json'
    with open(transforms_path) as f:
        transforms = json.load(f)
    annotation_poses.extend(np.array(transforms['transform_matrix']))

data_path = f'/home/slin/projects/nerf2nerf/scenes/scene_{scene_no}/{layout}/data.pt'
data = torch.load(data_path, map_location='cpu')
probs = data['probs']
xs = data['xs']
pts = xs[probs.squeeze(-1) > 0.9].numpy()

vis = Visualizer(show_frame=False, pt_size=3)
vis.add_trajectory(poses, cam_size=0.01)
vis.add_trajectory(annotation_poses, pose_spec=1, cam_size=0.01, color=(1, 0, 0))
vis.add_point_cloud(pts)
vis.show()
