import open3d as o3d
import numpy as np
import torch
from PIL import Image
import os
from tqdm import tqdm, trange

os.environ["PYTHONPATH"] = "data_preprocess/NLSPN_ECCV20"
from data_preprocess.NLSPN_ECCV20.src.config import args
from data_preprocess.NLSPN_ECCV20.src.model.nlspnmodel import NLSPNModel


def complete_depth(imgs, rgbs):
    ckpt_path = os.path.join('checkpoints', 'NLSPN_NYU.pt')
    checkpoint = torch.load(ckpt_path)
    args = checkpoint['args']
    args.test_only = True
    args.resume = True
    args.pretrain = ckpt_path

    model = NLSPNModel(args)
    net = model(args)
    net.cuda()
    key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
    net.eval()
    if key_u:
        print('Unexpected keys :')
        print(key_u)
    if key_m:
        print('Missing keys :')
        print(key_m)
        raise KeyError

    comp_list = []
    n = len(imgs)
    for i in range(n):
        rgb = rgbs[i]
        dep_sp = imgs[i]
        sample = {'rgb': rgb.cuda(), 'dep': dep_sp.cuda()}
        output = net(sample)
        comp_list.append(output)
    imgs = torch.cat(comp_list, dim=0)

    return imgs


def read_depth_imgs(base_dir):
    img_dir = os.path.join(base_dir, 'exported', 'depth')
    imgs = []
    file_names = os.listdir(img_dir)
    file_names = list(filter(lambda x: x.endswith(".png"), file_names))
    n = len(file_names)
    for i in range(n):
        img_path = os.path.join(img_dir, f"{i}.png")
        # img = o3d.io.read_image(img_path)
        img = np.array(Image.open(img_path)).astype(np.uint16)
        # img = o3d.geometry.Image(img)
        imgs.append(img)
    return imgs


def read_poses(base_dir):
    pose_dir = os.path.join(base_dir, 'exported', 'pose')
    poses = []
    file_names = os.listdir(pose_dir)
    file_names = list(filter(lambda x: x.endswith(".txt"), file_names))
    n = len(file_names)
    for i in range(n):
        pose_path = os.path.join(pose_dir, f"{i}.txt")
        pose = np.loadtxt(pose_path)
        poses.append(pose)
    return poses


def read_depth_intri(base_dir):
    intr_path = os.path.join(base_dir, 'exported', 'intrinsic', 'intrinsic_depth.txt')
    intr = np.loadtxt(intr_path)
    return intr[:3, :3]


def depth_to_pcd(imgs, poses, intri):
    depth_image = imgs[0]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1], height=depth_image.shape[0],
        fx=intri[0, 0], fy=intri[1, 1],
        cx=intri[0, 2], cy=intri[1, 2])
    merged_pcd = o3d.geometry.PointCloud()

    # base_dir = '/Users/bo233/Sci/workspace/ed-nerf/datasets/scannet/scene0101_04'
    # img_path = base_dir + "/exported/depth/{}.png"
    # pose_path = base_dir + "/exported/pose/{}.txt"
    # for i in trange(30):
    #     img = np.array(Image.open(img_path.format(i))).astype(np.uint16)
    #     pose = np.loadtxt(pose_path.format(i))
    #     pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(img), intrinsic, pose)
    #     merged_pcd += pcd

    for img, pose in tqdm(zip(imgs[:1000], poses[:1000])):
        # w2c is needed
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(img), intrinsic, np.linalg.inv(pose))
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intrinsic, pose)
        merged_pcd += pcd

    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)
    return merged_pcd


def main():
    base_dir = '/media/data/scannet_3dv/scans/scene0400_00'
    export_dir = os.path.join(base_dir, '')
    imgs = read_depth_imgs(base_dir)
    imgs = complete_depth(imgs)
    poses = read_poses(base_dir)
    intr = read_depth_intri(base_dir)
    pcd = depth_to_pcd(imgs, poses, intr)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(export_dir, 'pcd_from_depth.ply'), pcd)


if __name__ == "__main__":
    main()


# # 加载深度图
# depth_image = np.array(Image.open('/Users/bo233/Sci/workspace/ed-nerf/datasets/scannet/scene0241_01/exported/depth/109.png')).astype(np.uint16)

# # 定义相机内参和外参矩阵
# intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=500, fy=500, cx=depth_image.shape[1] // 2, cy=depth_image.shape[0] // 2)
# extrinsic = np.eye(4)

# # 将深度图转换为点云
# pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d.geometry.Image(depth_image), intrinsic=intrinsic, extrinsic=extrinsic, depth_trunc=3.0)

# # 可视化点云
# o3d.visualization.draw_geometries([pointcloud])
