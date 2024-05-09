import open3d as o3d
import numpy as np
from PIL import Image
import os
import sys
from tqdm import tqdm, trange


def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def read_depth_imgs(base_dir):
    img_dir = os.path.join(base_dir, 'depth')
    imgs = []
    file_names = os.listdir(img_dir)
    file_names = list(filter(lambda x: x.endswith(".png"), file_names))
    file_index = list(map(lambda x: int(x.split(".")[0]), file_names))
    file_index.sort()
    n = file_index[-1] + 1
    for i in range(n):
        img_path = os.path.join(img_dir, f"{i}.png")
        # img = o3d.io.read_image(img_path)
        img = np.array(Image.open(img_path)).astype(np.uint16)
        # img = o3d.geometry.Image(img)
        imgs.append(img)
    return imgs


def read_poses(base_dir):
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    pose_dir = os.path.join(base_dir, 'pose')
    poses = []
    file_names = os.listdir(pose_dir)
    file_names = list(filter(lambda x: x.endswith(".txt"), file_names))
    n = len(file_names)
    for i in range(n):
        pose_path = os.path.join(pose_dir, f"{i}.txt")
        pose = np.loadtxt(pose_path)
        if np.isnan(pose).any() or np.isinf(pose).any():
            print(f"pose {i} is nan or inf")
            continue
        pose = pose @ blender2opencv.T
        poses.append(pose)
    return poses


def read_depth_intri(base_dir):
    intr_path = os.path.join(base_dir, 'intrinsic', 'intrinsic_depth.txt')
    intr = np.loadtxt(intr_path)
    return intr[:3, :3]


def depth_to_pcd(imgs, poses, intri):
    depth_image = imgs[0]
    H, W = depth_image.shape
    K = intri[0, 0]
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     width=depth_image.shape[1], height=depth_image.shape[0],
    #     fx=intri[0, 0], fy=intri[1, 1],
    #     cx=intri[0, 2], cy=intri[1, 2])
    # merged_pcd = o3d.geometry.PointCloud()

    #n_empty = 10
    points_all = []
    #points_empty = []
    for img, pose in tqdm(zip(imgs[::2], poses[::2])):
        img = img / 1000.
        valid = img > 0
        valid_depth = img[valid]
        rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)
        norm = np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_d = rays_d / norm
        points = rays_o[valid] + rays_d[valid] * valid_depth[:, None]
        points_all.append(points)
        #print(f"valid points_num: {len(points)}")

        # d_mean = valid_depth.mean()
        # d_min = d_mean * 0.8
        # d_max = d_mean * 1.2
        # #print(f"d_min: {d_min}, d_max: {d_max}")
        # depth_empty = np.linspace(d_min, d_max, n_empty)
        # valid[:15, :] = True
        # valid[-15:, :] = True
        # valid[:, :15] = True
        # valid[:, -15:] = True
        # points = rays_o[~valid][None, ...] + rays_d[~valid][None, ...] * depth_empty[:, None, None]
        # points = points.reshape(-1, 3)
        # points_empty.append(points)
        #print(f"empty points_num: {len(points)}")

        # break
        # # w2c is needed
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(img), intrinsic, np.linalg.inv(pose))
        # # pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intrinsic, pose)
        # print(f"points_num: {len(pcd.points)}")
        # merged_pcd += pcd

    pcd_arr = np.concatenate(points_all, axis=0).reshape(-1, 3)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(pcd_arr)

    points_width = pcd_arr.max(axis=0) - pcd_arr.min(axis=0)
    voxel_size = points_width.min() / 256
    #voxel_size = points_width.min() / 128
    #voxel_size = points_width.min() / 64
    #voxel_size = points_width.min() / 32
    #voxel_size = points_width.min() / 16
    #voxel_size = points_width.min() / 8
    points_num = pcd_arr.shape[0]
    print(f"points_num: {points_num}, voxel_size: {voxel_size}")
    print(f"voxel_down_sample....")
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
    print("valid points_num: ", len(merged_pcd.points))
    # empty_pcd = np.concatenate(points_empty, axis=0).reshape(-1, 3)
    # empty_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(empty_pcd))
    # empty_pcd = empty_pcd.voxel_down_sample(voxel_size=voxel_size)
    # print("empty_points_num: ", len(empty_pcd.points))
    # merged_pcd += empty_pcd
    # print("merged points_num: ", len(merged_pcd.points))
    return merged_pcd


def main():
    scene = sys.argv[1]
    base_dir = f'/media/data/scannet_3dv/scans/{scene}'
    print(base_dir)
    # export_dir = os.path.join(base_dir, 'exported')
    imgs = read_depth_imgs(base_dir)
    print(len(imgs))
    poses = read_poses(base_dir)
    intr = read_depth_intri(base_dir)
    pcd = depth_to_pcd(imgs, poses, intr)
    print("pcd done")
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(base_dir, 'pcd.ply'), pcd)


if __name__ == "__main__":
    main()