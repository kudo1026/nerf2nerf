import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from nerfstudio.process_data.colmap_utils import (CameraModel, qvec2rotmat,
                                                  read_images_binary,
                                                  run_colmap)
from nerfstudio.process_data.hloc_utils import run_hloc
from scipy.spatial.transform import Rotation

from visualizer import Visualizer

# from utils.utils import (complete_transformation, compute_trans_diff,
#                          extract_colmap_pose, gen_hemispherical_poses)
# from utils.visualizer import Visualizer
# from view_renderer import ViewRenderer


def complete_transformation(T):
    """ completes T to be 4x4 """
    s = T.shape
    if s[-2:] == (4, 4):
        return T
    if isinstance(T, np.ndarray):
        return np.concatenate((T, np.tile(np.array([0, 0, 0, 1], dtype=T.dtype), s[:-2] + (1, 1))), axis=-2)
    return torch.cat((T, torch.tensor([0, 0, 0, 1], device=T.device).tile(s[:-2] + (1, 1))), dim=-2)


def extract_colmap_pose(colmap_im):
    rotation = qvec2rotmat(colmap_im.qvec)
    translation = colmap_im.tvec[:, None]
    w2c = complete_transformation(np.concatenate((rotation, translation), axis=1))
    c2w = np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system (spec 1) to nerfstudio's (spec 2)
    c2w = c2w @ np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])
    return c2w.astype(np.float32)


def decompose_sim3(T):
    """ T: 4x4 np array """
    G = T.copy()
    R = G[:3, :3]
    s = np.linalg.det(R)**(1 / 3)
    G[:3, :3] /= s
    return G, s


def compute_trans_diff(T1, T2):
    T = T2 @ np.linalg.inv(T1)
    G, s = decompose_sim3(T)
    R = Rotation.from_matrix(G[:3, :3])
    r = np.linalg.norm(R.as_rotvec()) / np.pi * 180
    t = np.linalg.norm(G[:3, 3])
    s = np.abs(np.log(s))
    return r, t, s


run_sfm = True
compute_trans = False
sfm_tool = 'hloc'

output_dir = Path('scenes/scene_6')
sfm_dir = output_dir / 'sfm'

if run_sfm:
    for matching_method in ["vocab_tree", "exhaustive", "sequential"]:
        for feature_type in ["sift", "superpoint_aachen", "superpoint_max", "superpoint_inloc", "r2d2", "d2net-ss", "sosnet", "disk"]:
            for matcher_type in ["superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"]:
                try:
                    sfm_dir = output_dir / f'sfm_{matching_method}_{feature_type}_{matcher_type}'
                    shutil.rmtree(sfm_dir, ignore_errors=True)
                    os.mkdir(sfm_dir)
                    for s in ['a', 'b']:
                        files = os.listdir(output_dir / s)
                        for f in files:
                            if not f.endswith('png') or f.startswith('rgb'):
                                continue
                            id = int(f.split('.')[0])
                            os.symlink((output_dir / s / f).absolute(), sfm_dir / f'{s}_{f}')
                    run_func = run_colmap if sfm_tool == 'colmap' else run_hloc
                    run_func(sfm_dir, sfm_dir, CameraModel.OPENCV, matching_method=matching_method, feature_type=feature_type, matcher_type=matcher_type)
                except:
                    pass

if compute_trans:
    images = read_images_binary(sfm_dir / 'sparse/0/images.bin')
    meta_A = torch.load(output_dir / 'a_in.pt', map_location='cpu')
    meta_B = torch.load(output_dir / 'b_in.pt', map_location='cpu')
    # poses of camAi_A
    poses_A = []
    # poses of camBi_B
    poses_B = []
    # poses of camCi_C
    poses_C = {}
    for im_data in images.values():
        fname = im_data.name
        id = fname[2:5]
        poses_C[fname] = extract_colmap_pose(im_data)
        if fname.startswith('a'):
            poses_A.append((fname, complete_transformation(meta_A[id].numpy()) @ np.array([[1, 0, 0, 0],
                                                                                          [0, -1, 0, 0],
                                                                                           [0, 0, -1, 0],
                                                                                           [0, 0, 0, 1]])))
        else:
            poses_B.append((fname, complete_transformation(meta_B[id].numpy()) @ np.array([[1, 0, 0, 0],
                                                                                          [0, -1, 0, 0],
                                                                                           [0, 0, -1, 0],
                                                                                           [0, 0, 0, 1]])))

    n = len(poses_A)
    print(f'Got {n} poses for A from SfM')
    s_lst = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            tAi_A = poses_A[i][1][:3, 3]
            tAj_A = poses_A[j][1][:3, 3]
            tAi_C = poses_C[poses_A[i][0]][:3, 3]
            tAj_C = poses_C[poses_A[j][0]][:3, 3]
            s_lst.append(np.linalg.norm(tAi_C - tAj_C) / np.linalg.norm(tAi_A - tAj_A))
    # s_AC = np.mean(s_lst)
    s_AC = np.median(s_lst)
    S_AC = np.diag((s_AC, s_AC, s_AC, 1)).astype(np.float32)
    T_AC_lst = [poses_C[pose[0]] @ S_AC @ np.linalg.inv(pose[1]) for pose in poses_A]
    # t_AC = sim3_log_map(torch.from_numpy(np.array(T_AC_lst))).mean(dim=0, keepdim=True)
    # T_AC = sim3_exp_map(t_AC)[0]
    # T_AC = np.mean(T_AC_lst, axis=0)
    T_AC = np.median(T_AC_lst, axis=0)
    u, _, vh = np.linalg.svd(T_AC[:3, :3])
    T_AC[:3, :3] = u * s_AC @ vh

    n = len(poses_B)
    print(f'Got {n} poses for B from SfM')
    s_lst = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            tBi_B = poses_B[i][1][:3, 3]
            tBj_B = poses_B[j][1][:3, 3]
            tBi_C = poses_C[poses_B[i][0]][:3, 3]
            tBj_C = poses_C[poses_B[j][0]][:3, 3]
            s_lst.append(np.linalg.norm(tBi_C - tBj_C) / np.linalg.norm(tBi_B - tBj_B))
    # s_BC = np.mean(s_lst)
    s_BC = np.median(s_lst)
    S_BC = np.diag((s_BC, s_BC, s_BC, 1)).astype(np.float32)
    T_BC_lst = [poses_C[pose[0]] @ S_BC @ np.linalg.inv(pose[1]) for pose in poses_B]
    # t_BC = sim3_log_map(torch.from_numpy(np.array(T_BC_lst))).mean(dim=0, keepdim=True)
    # T_BC = sim3_exp_map(t_BC)[0]
    # T_BC = np.mean(T_BC_lst, axis=0)
    T_BC = np.median(T_BC_lst, axis=0)
    u, _, vh = np.linalg.svd(T_BC[:3, :3])
    T_BC[:3, :3] = u * s_BC @ vh

    T_BA = np.linalg.inv(T_AC) @ T_BC
    np.save(output_dir / f'T_BA.npy', T_BA)
    print('pred trans\n', T_BA)

    with open(output_dir / 'gt_transform.json') as file:
        gt_poses = json.load(file)

    # defined in world?
    # with open(os.path.join(root, "object_point_clouds", opt.object_name + ".json")) as file:
    #     obj_pc = np.array(json.load(file))

    # model to A
    A = np.array(gt_poses['max_planck']['scene_a'])
    # model to B
    B = np.array(gt_poses['max_planck']['scene_b'])
    T_BA_gt = A @ np.linalg.inv(B)
    print('gt trans\n', T_BA_gt)
    r, t, s = compute_trans_diff(T_BA_gt, T_BA)
    print(f'rotation error {r:.4f}')
    print(f'translation error {t:.4f}')
    print(f'scale error {s:.4f}')

    viz = Visualizer(show_frame=True)
    # # gt
    # viz.add_trajectory([T_BA_gt], pose_spec=0, cam_size=0.1, color=(0, 0.7, 0))
    # # pred
    # viz.add_trajectory([T_BA], pose_spec=0, cam_size=0.1, color=(0.7, 0, 0.7))
    # auxiliary
    viz.add_trajectory(poses_C.values(), cam_size=0.05, color=(0, 0.7, 0))
    # viz.add_trajectory([poses_C[pose[0]] for pose in poses_A], cam_size=0.05, color=(0.7, 0, 0))
    # viz.add_trajectory(T_AC @ np.array([pose[1] for pose in poses_A]) @ np.linalg.inv(S_AC), cam_size=0.05, color=(0.7, 0.7, 0))
    # viz.add_trajectory([poses_C[pose[0]] for pose in poses_B], cam_size=0.05, color=(0, 0, 0.7))
    # viz.add_trajectory(T_BC @ np.array([pose[1] for pose in poses_B]) @ np.linalg.inv(S_BC), cam_size=0.05, color=(0, 0.7, 0.7))
    viz.show()
