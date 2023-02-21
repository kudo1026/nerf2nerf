# This script converts transforms.json from nerfstudio format to nerf2nerf format.

import json

import numpy as np


def convert_and_save(input_path, output_path):
    with open(input_path) as f:
        transforms = json.load(f)
    camera_angle_x = 2 * np.arctan(transforms['w'] / (transforms['fl_x'] * 2))
    frames = [{'transform_matrix': frame['transform_matrix']} for frame in transforms['frames']]
    with open(output_path, 'w') as f:
        json.dump({'camera_angle_x': camera_angle_x, 'frames': frames}, f, indent=2)


convert_and_save('/mnt/data/datasets/common_table/A/transforms.json', 'scenes/scene_0/transforms.json')
