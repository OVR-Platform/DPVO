import os
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def quaternions_to_yaw(quats_wxyz: np.ndarray) -> np.ndarray:
    if quats_wxyz.ndim != 2 or quats_wxyz.shape[1] != 4:
        raise ValueError("Expected quaternions with shape (N, 4) in wxyz order")

    w, x, y, z = quats_wxyz.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def compute_planar_deltas(timestamps: np.ndarray, positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    if len(timestamps) < 2:
        return np.empty((0, 5), dtype=np.float64)

    delta_xy = np.diff(positions[:, :2], axis=0)
    yaws = quaternions_to_yaw(quats_wxyz)
    delta_yaw = np.diff(np.unwrap(yaws))
    delta_dist = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    ts = timestamps[1:]

    return np.column_stack((ts, delta_xy[:, 0], delta_xy[:, 1], delta_yaw, delta_dist))

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):

    slam = None
    queue = Queue(maxsize=8)
    start_time = None
    frames_processed = 0

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        if frames_processed == 0:
            start_time = time.perf_counter()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

        frames_processed += 1

    reader.join()

    if frames_processed:
        elapsed = time.perf_counter() - start_time
        fps = frames_processed / elapsed if elapsed > 0 else float('inf')
        print(f"Processed {frames_processed} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--name', type=str, help='name your run', default=None)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--save_motion', action="store_true")
    parser.add_argument('--save_camera_poses', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)
    positions = poses[:, :3]
    quats_wxyz = poses[:, [6, 3, 4, 5]]
    trajectory = PoseTrajectory3D(positions_xyz=positions, orientations_quat_wxyz=quats_wxyz, timestamps=tstamps)

    run_name = args.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path("outputs") / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be stored in {output_root.resolve()}")

    if args.save_ply:
        ply_stem = output_root / "point_cloud"
        save_ply(str(ply_stem), points, colors)

    if args.save_colmap:
        colmap_dir = output_root / "colmap"
        save_output_for_COLMAP(str(colmap_dir), trajectory, points, colors, *calib)

    if args.save_trajectory:
        traj_path = output_root / "trajectory_tum.txt"
        file_interface.write_tum_trajectory_file(str(traj_path), trajectory)
        print(f"Trajectory saved to {traj_path}")

    if args.save_camera_poses:
        pose_path = output_root / "camera_poses.csv"
        pose_data = np.column_stack((tstamps, positions, quats_wxyz))
        header = "timestamp,x,y,z,qw,qx,qy,qz"
        np.savetxt(str(pose_path), pose_data, delimiter=",", header=header, comments="")
        print(f"Camera poses saved to {pose_path}")

    if args.save_motion:
        deltas_path = output_root / "motion_deltas.csv"
        delta_data = compute_planar_deltas(tstamps, positions, quats_wxyz)
        header = "timestamp,delta_x,delta_y,delta_yaw,delta_distance"
        np.savetxt(str(deltas_path), delta_data, delimiter=",", header=header, comments="")
        print(f"Planar deltas saved to {deltas_path}")

    if args.plot:
        plot_path = output_root / "trajectory_plot.pdf"
        plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {run_name}", filename=str(plot_path))


        

