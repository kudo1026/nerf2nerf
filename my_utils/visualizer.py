import numpy as np
import open3d as o3d

from .line_mesh import LineMesh


class Visualizer:
    def __init__(self, show_frame=False, pt_size=1) -> None:
        self.o3d_viz = o3d.visualization.Visualizer()
        self.o3d_viz.create_window()
        if show_frame:
            self.o3d_viz.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        self.pt_size = pt_size

    def add_trajectory(self, *poses, pose_spec=2, pose_type='c2w', cam_size=1, line_width=None, color=(0.5, 0.5, 0.5), connect_cams=False):
        """ pose_spec:
                0: x->right, y->front, z->up
                1: x->right, y->down, z->front
                2: x->right, y->up, z->back """
        if len(poses) == 1:
            Rs = [pose[:3, :3] for pose in poses[0]]
            ts = [pose[:3, 3] for pose in poses[0]]
        else:
            Rs, ts = poses
        if not hasattr(color[0], "__len__"):
            color = [color] * len(Rs)
        points = []
        lines = []
        colors = []
        for i, (R, t) in enumerate(zip(Rs, ts)):
            if pose_type == 'w2c':
                R = R.T
                t = -R @ t
            if pose_spec == 1:
                R = R @ np.array([[1, 0, 0],
                                  [0, 0, -1],
                                  [0, 1, 0]])
            elif pose_spec == 2:
                R = R @ np.array([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, -1, 0]])
            cam_points = [t]
            cam_points.append(R @ np.array([-1, 3, -1]) * cam_size + t)
            cam_points.append(R @ np.array([-1, 3, +1]) * cam_size + t)
            cam_points.append(R @ np.array([+1, 3, +1]) * cam_size + t)
            cam_points.append(R @ np.array([+1, 3, -1]) * cam_size + t)
            cam_points = np.stack(cam_points, axis=0)
            cam_lines = np.array([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)])
            points.extend(cam_points)
            if connect_cams and len(lines):
                cam_lines = np.vstack((cam_lines, [-5, 0]))
            lines.extend(i * 5 + cam_lines)
            colors.extend([color[i] for _ in range(len(cam_lines))])
        if line_width is None:
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
            self.o3d_viz.add_geometry(ls)
        else:
            lm = LineMesh(points, lines, colors, radius=line_width)
            for cylinder in lm.cylinder_segments:
                self.o3d_viz.add_geometry(cylinder)

    def add_point_cloud(self, pts, color=(0.5, 0, 0.5)):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if np.ndim(color) == 1:
            pc.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
        else:
            pc.colors = o3d.utility.Vector3dVector(color)
        self.o3d_viz.add_geometry(pc)

    def show(self):
        self.o3d_viz.get_render_option().point_size = self.pt_size
        self.o3d_viz.run()
