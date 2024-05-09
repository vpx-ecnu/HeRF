import torch
import numpy as np
from pykdtree.kdtree import KDTree as KDTree_py
from utils import timer

class KDTree(KDTree_py):
    def __init__(self, points, leafsize, **kwargs):
        super().__init__(points.detach().cpu().numpy(), leafsize, **kwargs)
        self.device = points.device
        self.init_points_index(points.shape[0])

    def init_points_index(self, points_shape):
        # N = self.n_points = points_shape
        D = self.kd_index_dim = int(pow(points_shape + 1, 1 / 3) + 1)
        points_index = np.meshgrid(range(D), range(D), range(D))
        points_index = np.stack(points_index, axis=-1).reshape(-1, 3)
        self.points_index = torch.tensor(points_index, device=self.device)

    def init_factor_tensor(self, feat_dim=32, device='cpu'):
        L = self.kd_index_dim
        P = L * L
        C = feat_dim
        # points_confs_line = torch.arange(C * L, device=device) / L
        # points_confs_plane = torch.arange(C * P, device=device) / P
        points_confs_line = torch.randn([C, L], device=device)
        points_confs_plane = torch.randn([C, L, L], device=device)
        points_confs_line = points_confs_line.reshape(1, C, L, 1)
        points_confs_plane = points_confs_plane.reshape(1, C, L, L)
        return points_confs_line, points_confs_plane


class KDCache():
    def __init__(self, kdtree, cache_bbox, cache_grid_num, k_query, dist_thre=15.0):
        self.cache_dist = None
        self.cache_index = None
        self.cache_confs = None

        self.device = kdtree.device
        self.k_query = k_query
        self.cache_bbox = cache_bbox.to(self.device)
        self.cache_grid_num = torch.tensor(cache_grid_num, device=self.device)
        self.cache_grid_size = (self.cache_bbox[1] - self.cache_bbox[0]) / self.cache_grid_num
        self.dist_thre = dist_thre * min(self.cache_grid_size)
        grid_coord = self.build_grid_coordinate()
        self.cache = self.build(grid_coord, kdtree)

    def build_grid_coordinate(self):
        x = np.arange(
            self.cache_bbox[0][0],
            self.cache_bbox[1][0],
            self.cache_grid_size[0],
            dtype=np.float32
        )
        y = np.arange(
            self.cache_bbox[0][1],
            self.cache_bbox[1][1],
            self.cache_grid_size[1],
            dtype=np.float32
        )
        z = np.arange(
            self.cache_bbox[0][2],
            self.cache_bbox[1][2],
            self.cache_grid_size[2],
            dtype=np.float32
        )
        X, Y, Z = np.meshgrid(x, y, z)
        coord = np.stack([Y[..., None], X[..., None], Z[..., None]], axis=-1)
        return coord + self.cache_grid_size.detach().cpu().numpy() / 2.0

    def build(self, points, kdtree):
        H, W, D = points.shape[:3]
        points = points.reshape(H * W * D, 3)

        dist, idx = kdtree.query(points, k=self.k_query)
        self.cache_dist = torch.tensor(dist, device=self.device)
        self.cache_self_index = torch.tensor(idx.astype(np.float32), dtype=torch.long)
        self.cache_tree_index = kdtree.points_index[self.cache_self_index]

    def index_3d_to_index_1d(self, index):
        X, Y, Z = self.cache_grid_num
        index = index[:, 0] * Y * Z + index[:, 1] * Z + index[:, 2]
        return index.long()

    def get_neighbor_indices(self, index):
        x, y, z = index[:, 0], index[:, 1], index[:, 2]
        indices = []
        for i, j, k in [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
            x_, y_, z_ = x + i, y + j, z + k
            index = torch.stack([x_, y_, z_], dim=-1)
            indices.append(index)
        return torch.stack(indices, dim=1)

    def valid_mask(self, index):
        x_valid = torch.logical_and(index[:, 0] >= 0, index[:, 0] < self.cache_grid_num[0])
        y_valid = torch.logical_and(index[:, 1] >= 0, index[:, 1] < self.cache_grid_num[1])
        z_valid = torch.logical_and(index[:, 2] >= 0, index[:, 2] < self.cache_grid_num[2])
        a_valid = torch.logical_and(torch.logical_and(x_valid, y_valid), z_valid)
        return a_valid

    def clip_index(self, index):
        index[..., 0] = torch.clip(index[..., 0], 0, self.cache_grid_num[0] - 1)
        index[..., 1] = torch.clip(index[..., 1], 0, self.cache_grid_num[1] - 1)
        index[..., 2] = torch.clip(index[..., 2], 0, self.cache_grid_num[2] - 1)
        return index

    @timer
    def query_cache(self, samples, factor_tensor_plane, factor_tensor_line, matMode, vecMode, detach):

        index = torch.floor((samples - self.cache_bbox[0]) / self.cache_grid_size).int()
        near_index = self.get_neighbor_indices(index)
        N, J, _ = near_index.shape
        M, K = self.cache_dist.shape
        near_index_valid = self.valid_mask(near_index.reshape(-1, 3))
        near_index = self.clip_index(near_index)
        near_index = self.index_3d_to_index_1d(near_index.reshape(-1, 3))
        # near_index_3d = near_index_3d.reshape(N * J, 3)

        near_grid_dist = self.cache_dist[near_index]
        near_grid_index = self.cache_tree_index[near_index].reshape(-1, 3)
        near_grid_dist[~near_index_valid] = 10e5

        # sample_line = factor_tensor_line[near_grid_index[:, :, 2]]
        # sample_plane = factor_tensor_plane[near_grid_index[:, :, 0], near_grid_index[:, :, 1]]

        coordinate_plane = torch.stack((near_grid_index[..., matMode[0]],
                                        near_grid_index[..., matMode[1]],
                                        near_grid_index[..., matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((near_grid_index[..., vecMode[0]],
                                       near_grid_index[..., vecMode[1]],
                                       near_grid_index[..., vecMode[2]])).view(3, -1, 1, 1)

        if detach:
            coordinate_plane = coordinate_plane.detach()
            coordinate_line = coordinate_line.detach()

        invalid_mask = near_grid_dist > self.dist_thre
        near_grid_dist_clip = torch.clip(near_grid_dist, 1e-3, self.dist_thre)
        weight = 1.0 / (near_grid_dist_clip + 10e-8)
        weight[invalid_mask] = 0.0
        weight = weight.reshape(1, N, K*J)
        weight_sum = (torch.sum(weight, dim=-1) + 10e-8)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(factor_tensor_line)):
            # sample_line_N = coordinate_plane.reshape(N, K * J, -1)
            # sample_plane_N = coordinate_line.reshape(N, K*J, -1)
            # near_grid_dist_N = near_grid_dist.reshape(N, K*J, -1)
            sample_line, sample_plane = self.interpolate_feature(weight, weight_sum,
                                                                 coordinate_plane[idx_plane],
                                                                 coordinate_line[idx_plane],
                                                                 factor_tensor_plane[idx_plane],
                                                                 factor_tensor_line[idx_plane],
                                                                 N, M, J, K,
                                                                 )
            plane_coef_point.append(sample_plane.reshape(-1, N))
            line_coef_point.append(sample_line.reshape(-1, N))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        return plane_coef_point, line_coef_point

    def interpolate_feature(self, weight, weight_sum, coordinate_plane, coordinate_line,
                            factor_tensor_plane, factor_tensor_line, N, M, J, K):

        coordinate_plane = coordinate_plane.reshape(N * K * J, -1)
        coordinate_line = coordinate_line.reshape(N * K * J, -1)
        sample_plane = factor_tensor_plane[0, :, coordinate_plane[:, 0], coordinate_plane[:, 1]].reshape(-1, N, K * J)
        sample_line = factor_tensor_line[0, :, coordinate_line[:, 0]].reshape(-1, N, K * J)
        sample_line = torch.sum(sample_line * weight, dim=-1) / weight_sum
        sample_plane = torch.sum(sample_plane * weight, dim=-1) / weight_sum
        return sample_line, sample_plane
