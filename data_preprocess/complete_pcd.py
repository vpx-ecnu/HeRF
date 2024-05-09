import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


def read_points_from_path(points_path):
    plydata = PlyData.read(points_path)
    x, y, z = (
        torch.as_tensor(
            plydata.elements[0].data["x"].astype(np.float32),
            device="cpu",
            dtype=torch.float32,
        ),
        torch.as_tensor(
            plydata.elements[0].data["y"].astype(np.float32),
            device="cpu",
            dtype=torch.float32,
        ),
        torch.as_tensor(
            plydata.elements[0].data["z"].astype(np.float32),
            device="cpu",
            dtype=torch.float32,
        ),
    )
    points_xyz = torch.stack([x, y, z], dim=-1)
    del plydata
    # ranges: -10.0 -10.0 -10.0 10.0 10.0 10.0
    # if self.opt.ranges[0] > -99.0:s
    #     ranges = torch.as_tensor(self.opt.ranges, device=points_xyz.device, dtype=torch.float32)
    #     mask = torch.prod(torch.logical_and(points_xyz >= ranges[None, :3], points_xyz <= ranges[None, 3:]), dim=-1) > 0
    #     points_xyz = points_xyz[mask]

    mask = torch.isnan(points_xyz[:, 0])
    points_xyz = points_xyz[~mask]
    mask = torch.isnan(points_xyz[:, 1])
    points_xyz = points_xyz[~mask]
    mask = torch.isnan(points_xyz[:, 2])
    points_xyz = points_xyz[~mask]
    return points_xyz

def vis_depths(depths):
    depth = depths[0]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    # 转换为0-255范围的值
    depth_8bit = (depth_normalized * 255).to(torch.uint8)

    # 转换为PIL图像
    depth_image = Image.fromarray(depth_8bit.numpy())

    # 存储图像
    depth_image_path = 'debug/depth_image.png'
    depth_image.save(depth_image_path)


def project_points_to_depth(pcd, K, extrinsic, wh):
    """
    Project points to a depth map using a PyTorch3D renderer.

    Args:
        pcd (torch.Tensor): Point cloud as N x 3 tensor.
        intrinsic (torch.Tensor): Camera intrinsic matrix as 3 x 3 tensor.
        extrinsic (torch.Tensor): Camera extrinsic matrix as n x 4 x 4 tensor.

    Returns:
        torch.Tensor: Depth map.
    """

    # Create a FoVPerspectiveCameras object
    R = extrinsic[:3, :3][None]  # Rotation matrix
    T = extrinsic[:3, 3][None]   # Translation vector
    cameras = FoVPerspectiveCameras(device="cuda", R=R, T=T, K=K)

    # Create a Pointclouds object
    feat = torch.ones_like(pcd)
    point_cloud = Pointclouds(points=[pcd], features=[feat])

    # Define the rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=(wh[0], wh[1]),  # Specify the image size
        radius=0.5,         # The radius of each point in NDC
        points_per_pixel=10  # Number of points per pixel
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(point_cloud)

    return fragments.zbuf[:, :, :, 0]

def project_points_to_depth_list(pcd, intrinsic, extrinsics, wh):
    depth_list = []
    K = torch.eye(4)
    K[:3, :3] = intrinsic
    K = K[None]

    pcd = pcd.to('cuda')
    K = K.to('cuda')
    extrinsics = extrinsics.to('cuda')

    for extrinsic in extrinsics:
        depth = project_points_to_depth(pcd, K, extrinsic, wh)
        depth_list.append(depth)

    depths = torch.cat(depth_list, dim=0)
    return depths