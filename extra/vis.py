import numpy
import open3d as o3d
import torch

debug_path = "/Users/minisal/Desktop/debug/"

color = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1],
         [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
         [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.5, 0.5, 0.5], [0.5, 0.5, 1], [0.5, 1, 0.5],
         [1, 0.5, 0.5], [0.5, 1, 1], [1, 0.5, 1],
         [1, 1, 0.5], [0.2, 1, 1], [1, 0.2, 1],
         [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2],
         [0.2, 0.2, 0], [0.2, 0, 0.2], [0, 0.2, 0.2],
         ]

# file_path = "/Users/minisal/Desktop/debug/input.ply"
# input = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/active.ply"
# active = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/index.ply"
# index = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/reindex.ply"
# reindex = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/active_ups.ply"
# active_ups = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/index_ups.ply"
# index_ups = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/reindex_ups.ply"
# reindex_ups = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/query_feat.ply"
# query_feat = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/voxel_feat.ply"
# voxel_feat = o3d.io.read_point_cloud(file_path)
# file_path = "/Users/minisal/Desktop/debug/query.ply"
# query = o3d.io.read_point_cloud(file_path)
file_path = "/Users/minisal/Desktop/debug/scene0241_01_vh_clean.ply"
gt = o3d.io.read_point_cloud(file_path)
o3d.visualization.draw_geometries([gt])

todo = 1.
def o3d_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if isinstance(colors, list):
            colors = numpy.array([colors] * points.shape[0])
        else:
            colors = colors.detach().cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float).reshape(-1, 3))
    return pcd

def load(file):
    return torch.load(debug_path+file, map_location=torch.device('cpu'))
def visualize_point_cloud(point_cloud, camera_poses):
    """
    Visualizing point clouds and camera extrinsics.
    Args:
        point_cloud: numpy.array，[N, 3]
        camera_poses: numpy.array，[N, 4, 4]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for pose in camera_poses:
        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.1, resolution=4)
        cone.transform(pose)
        vis.add_geometry(cone)
    vis.get_render_option().point_size = 2
    vis.get_view_control().set_zoom(0.8)
    vis.run()
    vis.destroy_window()


# input = load("/input.pt").numpy()
# active = load("/active.pt").reshape(-1).numpy()
# index = load("/index.pt").reshape(-1, 3).numpy()
# reindex = load("/reindex.pt").reshape(-1, 3).numpy()
# reindex_batch = load("/reindex_batch.pt").reshape(-1, 3).numpy()
# voxel_coord = load("/voxel_coord.pt").reshape(-1, 3).numpy()
#plane_coef_point_full = load("/plane_coef_point_full.pt")
#line_coef_point_full = load("/line_coef_point_full.pt")
# samples = load("/samples.pt")
# samples_active = load("/samples_active.pt")[0]
# samples_from_index = load("/samples_from_index.pt")
# batch_points = load("/batch_points.pt")
# xyz_sampled = load("/xyz_sampled.pt")
# sigma_feature = load("/sigma_feature.pt")
# xyz_sampled_app = load("/xyz_sampled_app.pt")
# app_features = load("/app_features.pt")
#index_block = load("/index_block.pt")
#index_block_active = load("/index_block_active.pt")
#index_sparse_near = load("/index_sparse_near.pt")
#index_block_1d = load("/index_block_1d.pt")
#weight = load("/weight.pt")
# # -------------------------------------------
points_per_bbox = load("/points_per_bbox.pt")
points_per_bbox = [o3d_pcd(points_per_bbox[i], color[i]) for i in range(len(points_per_bbox))]
o3d.visualization.draw_geometries(points_per_bbox)
# -------------------------------------------
reindex_batch_1d_all = load("/reindex_batch_1d_all.pt")
color = reindex_batch_1d_all[:, :, -1].view(-1, 1) / reindex_batch_1d_all.max()
zero = torch.zeros_like(color)
color = torch.cat([color, zero, zero], dim=1)
reindex_batch_1d_all = o3d_pcd(reindex_batch_1d_all[:, :, :3].view(-1, 3), color)
o3d.visualization.draw_geometries(points_per_bbox+[reindex_batch_1d_all])
o3d.visualization.draw_geometries([points_per_bbox])
# # -------------------------------------------

# active_data = load("/active_data.pt")
# active = active_data['active'][0]
# active_ups = active_data['active_ups'][0]
# active_coord = active_data['active_coord']
# active_color = torch.zeros_like(active_coord)
# active_color[active.bool()] = torch.tensor(color[0], dtype=torch.float32)
# active_color[active_ups.bool()] = torch.tensor(color[1], dtype=torch.float32)
# both = torch.logical_and(active.bool(), active_ups.bool())
# active_color[both] = torch.tensor(color[2], dtype=torch.float32)
# valid = torch.logical_or(active, active_ups)
# # valid_add = torch.where((active_ups + active) >= 1.)[0]
#
# # valid = torch.where(active_ups)
# active = o3d_pcd(active_coord[valid], active_color[valid])
# o3d.visualization.draw_geometries([active])
# -------------------------------------------

todo = 1.
# thre = 0.2
# xyz_sampled = xyz_sampled[sigma_feature > thre]
# sigma_feature = sigma_feature[sigma_feature > thre].view(-1, 1).repeat(1, 3)
# sigma_feature = 1. - sigma_feature / sigma_feature.max()
# xyz_sampled = o3d_pcd(xyz_sampled, sigma_feature)
#xyz_sampled_app = o3d_pcd(xyz_sampled_app, [1, 0, 0])
#app_features = o3d_pcd(app_features, [0, 0, 0])

#pcd_sample = o3d_pcd(samples)
# o3d.visualization.draw_geometries([pcd_sample])

# pcd_sample_active = o3d_pcd(samples_active)
# o3d.visualization.draw_geometries([pcd_sample_active])

# p = voxel_coord[active]
# c = index[active]
# pcd_index = o3d_pcd(p, c)
# o3d.visualization.draw_geometries([pcd_index, pcd_sample_active])


# pcd_input = o3d_pcd(input)
#
# c = numpy.zeros_like(voxel_coord)
# c[active, :] = 1
# pcd_active = o3d_pcd(voxel_coord, c)
# o3d.visualization.draw_geometries([pcd_active])
#


pcd_reindex = o3d_pcd(voxel_coord[active], reindex[active]/reindex.max())
pcd_reindex_batch = o3d_pcd(voxel_coord[active], reindex_batch/reindex_batch.max())
o3d.visualization.draw_geometries([pcd_reindex])

todo = 1.