import open3d as o3d
import numpy as np
from PIL import Image
import os
import sys
import json
import cv2
from tqdm import tqdm, trange


def read_depth_imgs(base_dir):
    img_dir = os.path.join(base_dir, 'test')
    imgs = []
    file_names = os.listdir(img_dir)
    file_names = list(filter(lambda x: "depth" in x, file_names))
    file_index = list(map(lambda x: int(x.split("_")[1]), file_names))
    post_fix = file_names[0][-8:-4]
    file_index.sort()
    n = file_index[-1] + 1
    for i in range(n):
        img_path = os.path.join(img_dir, f"r_{i}_depth_{post_fix}.png")

        depth = cv2.imread(img_path).astype(float)[:, :, 0]
        invalid = depth == 0
        depth = 8 * (255.0 - depth) / 255
        depth[invalid] = 0.

        imgs.append(depth)
    return imgs


def read_poses(base_dir):
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    pose_file = os.path.join(base_dir, 'transforms_test.json')
    with open(pose_file, 'r') as f:
        meta = json.load(f)

    poses = []
    idxs = list(range(0, len(meta["frames"])))
    for i in idxs:
        frame = meta["frames"][i]
        pose = frame["transform_matrix"]
        pose = np.array(pose)
        #pose = pose @ blender2opencv
        poses.append(pose)
    return poses


def read_depth_intri(base_dir, H, W):
    meta_file = os.path.join(base_dir, 'transforms_test.json')
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    #print(focal)
    return focal

def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def depth_to_pcd(imgs, poses, focal):
    depth_image = imgs[0]
    H, W = depth_image.shape
    K = focal
    points_all = []

    for img, pose in tqdm(zip(imgs[::2], poses[::2])):
        rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)
        norm = np.linalg.norm(rays_d, axis=2, keepdims=True)
        valid = img > 0
        points = rays_o + rays_d * img[:, :, None] / norm
        points = points[valid]
        points = points.reshape(-1, 3)
        points_all.append(points)

    points_all = np.concatenate(points_all, axis=0).reshape(-1, 3)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(points_all)
    #points_width = points_all.max(axis=0) - points_all.min(axis=0)
    #points_width = np.abs(points_width)
    #voxel_size = points_width.max() / 512.
    #print("voxel_size", voxel_size)
    #merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
    return merged_pcd


def main():
    scene = sys.argv[1]
    base_dir = f'/media/data/yxy/ed-nerf/data/nerf/nerf_synthetic/{scene}'
    print(base_dir)
    imgs = read_depth_imgs(base_dir)
    print(len(imgs))
    poses = read_poses(base_dir)
    depth_image = imgs[0]
    H, W = depth_image.shape
    intr = read_depth_intri(base_dir, H, W)
    pcd = depth_to_pcd(imgs, poses, intr)
    print("pcd done")
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(base_dir, 'pcd.ply'), pcd)


if __name__ == "__main__":
    main()