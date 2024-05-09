import math
import torch
import torch.nn.functional as F
import numpy as np

from utils import timer

# import pickle


DEBUG_SAVE = False
debug_path = "/media/data/yxy/ed-nerf/logs/tensoir/log_original_blender/debug"
# debug_path = "/Users/minisal/Desktop/debug"


class Block:
    def __init__(
        self,
        points=None,
        N_voxels_every_dim=128,
        bbox_edge_scale=0.01,
        N_dilate_kernel=3,
        N_voxels_in_batch=8,
        n_block_per_axis=1,
        reindex_start=1,
        padding_tensor_on=True,
        sparse_voxel_on=True
    ):
        self.N_block_size = 1
        self.N_dilate_kernel = N_dilate_kernel
        self.N_voxels_in_batch = N_voxels_in_batch
        self.N_voxels_every_dim = N_voxels_every_dim
        self.N_block_per_axis = n_block_per_axis
        self.padding_tensor_on = padding_tensor_on
        self.sparse_voxel_on = sparse_voxel_on

        # self.bbox = None
        self.voxel_size = None
        # self.voxel_coord = None
        self.active = None
        self.index = None
        self.active_batch_1d = None
        self.reindex_batch = None
        self.reindex_batch_1d = None
        self.reindex_batch_1d_global = None
        self.padding_coord_edge = None
        self.padding_coord_corner = None
        self.voxel_size_old = 0.

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

        if points is None:
            return

        assert N_voxels_in_batch <= N_voxels_every_dim

        self.points = points
        self.device = points.device
        points = points.unsqueeze(0)
        self._init_voxel_params(points, bbox_edge_scale)
        self.build(points, reindex_start)

    def _init_voxel_params(self, points, bbox_edge_scale):
        # TODO compatible when sparse_voxel_on is False
        point_min = points.min(dim=1)[0]
        point_max = points.max(dim=1)[0]
        bbox_size = point_max - point_min
        point_min -= bbox_size * bbox_edge_scale
        point_max += bbox_size * bbox_edge_scale
        bbox_size = point_max - point_min
        voxel_size = bbox_size / self.N_voxels_every_dim
        bbox_min = point_min + voxel_size / 2
        bbox_max = point_max - voxel_size / 2
        bbox = torch.stack([point_min, point_max], dim=1)
        self.voxel_size = voxel_size
        self.voxel_bbox = torch.stack([bbox_min, bbox_max], dim=1)
        self.bbox = bbox

    def index_3d_to_index_1d(self, index, voxel_num):
        index = index.long()
        V = voxel_num
        index = index[:, :, 0] * V * V + index[:, :, 1] * V + index[:, :, 2]
        return index

    def index_1d_to_index_3d(self, index, voxel_num):
        index = index.long()
        V = voxel_num
        index_x = torch.div(index, V * V, rounding_mode="floor")
        index_y = torch.div(index - index_x * V * V, V, rounding_mode="floor")
        index_z = index - index_x * V * V - index_y * V
        index = torch.stack([index_x, index_y, index_z], dim=-1).long()
        return index

    def get_index_from_points(self, points, dtype="long"):
        voxel_size = self.voxel_size.view(self.N_block_size, 1, 3)
        index = (points - self.voxel_bbox[:, :1]) / voxel_size
        if dtype == "long":
            index = index.round().long()
        return index

    def _set_active(self, index):
        V = self.N_voxels_every_dim
        B = self.N_block_size
        size = [B, V * V * V]
        active = torch.zeros(size, device=self.device)
        index_1d = self.index_3d_to_index_1d(index, V)
        index_1d_valid = self.valid_mask(index)
        if index_1d.shape[0] == active.shape[0] == 1:
            index_1d = index_1d[index_1d_valid]
            active[0, index_1d] = 1
        else:
            active_list = torch.split(active, 1, dim=0)

            for i in range(B):
                i_index = index_1d[i]
                i_index = i_index[index_1d_valid[i]]
                # i_index = i_index[i_index >= 0]
                i_index = torch.unique(i_index)
                active_list[i][0, i_index] = 1
            active = torch.cat(active_list, dim=0)

        # active_dilate = F.max_pool3d(
        #     active.view(B, 1, V, V, V),
        #     kernel_size=self.N_dilate_kernel,
        #     padding=int(self.N_dilate_kernel // 2),
        #     stride=1,
        # )[:, 0].view(B, -1)
        # active = active_dilate

        if self.active is not None:
            V_ = math.ceil(pow(self.active.shape[1], 1 / 3))
            active_ups = torch.nn.functional.interpolate(
                self.active.reshape(B, 1, V_, V_, V_).float(),
                size=[V, V, V],
                mode="trilinear",
                # mode="nearest",
                align_corners=True,
            ).view(B, V * V * V)

            # DEBUG
            # self._set_voxel_coord()
            # active_data = {"active": active,
            #                "active_ups": active_ups,
            #                "active_coord": self.voxel_coord}
            # torch.save(active_data, debug_path + '/active_data.pt')

            active = torch.logical_or(active.bool(), active_ups.bool())
            # active = active_ups.bool()

        self.active = active.bool().view(size)
        if not self.sparse_voxel_on:
            self.active[:] = True

    def _set_active_batch(self, index):
        self.N_batch_num = int(self.N_voxels_every_dim / self.N_voxels_in_batch)
        batch_index = torch.div(index, self.N_voxels_in_batch, rounding_mode="floor")
        active_batch_1d = self.index_3d_to_index_1d(batch_index, self.N_batch_num)
        active_batch_1d = active_batch_1d.unique(dim=-1)
        if self.active_batch_1d is not None:
            active_batch_1d_old = torch.isin(active_batch_1d, self.active_batch_1d)
            active_batch_1d_new = active_batch_1d[~active_batch_1d_old]
            active_batch_1d = torch.cat([active_batch_1d, active_batch_1d_new])
        self.active_batch_1d = active_batch_1d

    def _set_index(self):
        index = [
            torch.arange(0, self.N_voxels_every_dim, device=self.device)
            for i in range(3)
        ]
        index = torch.stack(torch.meshgrid(index), dim=-1).reshape(-1, 3)
        self.index = index

    def _set_voxel_coord(self):
        if self.index is None:
            self._set_index()
        coord = self.index * self.voxel_size + self.voxel_bbox[:, 0]
        self.voxel_coord = coord

    def _set_reindex_batch(self, reindex_start):
        B = self.N_block_size
        N = self.active.shape[-1]
        N_active = torch.sum(self.active, dim=1).item()
        N_active_batch = self.active_batch_1d.shape[-1]
        print(f"active voxels: {N_active} , sparsity: {N_active / N}")
        print(f"active batch voxels: {N_active_batch}")
        assert self.active.shape[0] == self.active_batch_1d.shape[0] == B == 1

        if self.reindex_batch_1d is None:
            reindex_batch_1d = torch.arange(0, N_active_batch, device=self.device)
            reindex_batch_1d = reindex_batch_1d + reindex_start
        else:
            raise NotImplementedError
        # if self.reindex_batch_1d is not None:
        #     raise NotImplementedError
        #     reindex_batch_1d_left = torch.arange(0, N_reindex_batch, device=self.device)
        #     reindex_batch_1d_on = torch.isin(
        #         reindex_batch_1d_left, self.reindex_batch_1d
        #     )
        #     reindex_batch_1d_left = reindex_batch_1d_left[~reindex_batch_1d_on]
        #     N_active_batch_new = N_active_batch - self.reindex_batch_1d.shape[0]
        #     reindex_indices_new = torch.arange(
        #         0, N_active_batch_new, device=self.device
        #     )
        #     reindex_indices_new = (
        #         reindex_indices_new
        #         * reindex_batch_1d_left.shape[0]
        #         / N_active_batch_new
        #     ).long()
        #     reindex_batch_1d_new = reindex_batch_1d_left[reindex_indices_new]
        #     reindex_batch_1d = torch.cat([self.reindex_batch_1d, reindex_batch_1d_new])
        # print("reindex_batch_1d", reindex_batch_1d.shape[0])
        # print("N_active_batch", N_active_batch)
        # assert reindex_batch_1d.shape[0] == N_active_batch

        # N_batch = int(self.N_voxels_in_batch**3)
        N_batch = int(self.N_batch_num**3)
        reindex_batch_all = torch.zeros((N_batch, 1), device=self.device).long()
        reindex_batch_all[self.active_batch_1d] = reindex_batch_1d[:, None]
        self.reindex_batch_1d_all = reindex_batch_all.view(1, -1)
        self.reindex_batch_1d = reindex_batch_1d
        # generate padding tensor coordinate

    def _set_padding_coord(self):
        if not self.padding_tensor_on:
            return

        # select group origin by re-index
        # if self.padding_coord_edge is not None:
        if False:

            # padding_coord_corner = self.padding_coord_corner[0]

            edge = self.padding_coord_edge[0]
            grid = torch.arange(0, self.N_voxels_in_batch, device=self.device).view(1, 1, -1, 1)
            zero = torch.zeros_like(grid)
            grid = grid / self.N_voxels_in_batch * 2 - 1
            grid = torch.cat([zero, grid], dim=-1)

            N, _, _, H, C = edge.shape
            edge = edge.permute(0, 1, 2, 4, 3).reshape(N*3*4, C, H, 1)
            grid = grid.expand(N*3*4, 1, -1, 2)
            edge = torch.nn.functional.grid_sample(edge, grid, padding_mode='border')
            edge = edge.permute(0, 2, 1, 3).reshape(1, N, 3, 4, -1, C)
            bias = self.voxel_size - self.voxel_size_old
            # todo corner bias

            corner = self.padding_coord_corner[0]

            for i in range(3):
                mat = self.matMode[i]
                vec = self.vecMode[i]

                # corner[:, i, :, mat] -= bias[:, mat]
                # corner[:, i, 4, vec] += bias[:, vec]
                corner[:, i, 0, mat[0]] -= bias[:, mat[0]]
                corner[:, i, 0, mat[1]] -= bias[:, mat[1]]
                corner[:, i, 1, mat[0]] -= bias[:, mat[0]]
                corner[:, i, 1, mat[1]] += bias[:, mat[1]]
                corner[:, i, 2, mat[0]] += bias[:, mat[0]]
                corner[:, i, 2, mat[1]] -= bias[:, mat[1]]
                corner[:, i, 3, mat[0]] += bias[:, mat[0]]
                corner[:, i, 3, mat[1]] += bias[:, mat[1]]
                corner[:, i, 4, mat[0]] -= bias[:, mat[0]]
                corner[:, i, 4, mat[1]] -= bias[:, mat[1]]
                corner[:, i, 4, vec] += bias[:, vec]

            self.padding_coord_edge = edge.detach()
            self.padding_coord_corner = corner.detach()[None, ...]
            # index_3d_float = self.get_index_from_points(edge, dtype="float")
            # index_3d_float = self.get_index_from_points(corner, dtype="float")
            # self.get_index_from_points(corner.view(1, -1, 4)[..., :3], dtype="float").view(8, 3, 5, 3)[0, 0]
            # self.get_index_from_points(self.padding_coord_edge.view(1, -1, 4)[..., :3], dtype="float").view(8, 3, 4, 37, 3)[0, 0, :, 0, :]

            # V = self.N_voxels_in_batch
            # B = self.N_block_size
            # origin = [
            #     coord[i, self.active_batch_1d[i][0], :]
            #     for i in range(self.N_block_size)
            # ]
            #
            # padding_step_edge = padding_step_edge.view(
            #     1, 3, 4, V, 3
            # )
            # padding_step_corner = padding_step_corner.view(1, 3, 5, 3)
            #
            # padding_coord_edge = [
            #     origin[i].view(-1, 1, 1, 1, 3) + padding_step_edge * r[i]
            #     for i in range(B)
            # ]
            # padding_coord_corner = [
            #     origin[i].view(-1, 1, 1, 3) + padding_step_corner * r[i]
            #     for i in range(B)
            # ]
            #
            # padding_coord_edge = torch.cat(padding_coord_edge, dim=0)[None, ...]
            # padding_coord_corner = torch.cat(padding_coord_corner, dim=0)[None, ...]

        else:
            # r = self.voxel_size * self.N_batch_num

            padding_step_edge = torch.zeros((3, 4, self.N_voxels_in_batch, 3), device=self.device)
            padding_step_corner = torch.zeros((3, 6, 3), device=self.device)
            padding_step_edge_inner = torch.zeros((3, 4, self.N_voxels_in_batch, 3), device=self.device)
            padding_step_corner_inner = torch.zeros((3, 6, 3), device=self.device)
            for i in range(3):
                mat = self.matMode[i]
                vec = self.vecMode[i]
                increase = torch.arange(0, self.N_voxels_in_batch,device=self.device)

                padding_step_edge[i, 0, :, mat[0]] -= 1 #- 1
                padding_step_edge[i, 0, :, mat[1]] += increase
                padding_step_edge[i, 1, :, mat[0]] += self.N_voxels_in_batch #- 1  # todo todo todo, all the points should have value
                padding_step_edge[i, 1, :, mat[1]] += increase
                padding_step_edge[i, 2, :, mat[0]] += increase
                padding_step_edge[i, 2, :, mat[1]] -= 1 #- 1
                padding_step_edge[i, 3, :, mat[0]] += increase
                padding_step_edge[i, 3, :, mat[1]] += self.N_voxels_in_batch #- 1

                # padding_step_edge[i, 0, :, mat[0]] -= 1  # - 1
                padding_step_edge_inner[i, 0, :, mat[1]] += increase
                padding_step_edge_inner[i, 1, :, mat[0]] += self.N_voxels_in_batch - 1
                padding_step_edge_inner[i, 1, :, mat[1]] += increase
                padding_step_edge_inner[i, 2, :, mat[0]] += increase
                # padding_step_edge[i, 2, :, mat[1]] -= 1  # - 1
                padding_step_edge_inner[i, 3, :, mat[0]] += increase
                padding_step_edge_inner[i, 3, :, mat[1]] += self.N_voxels_in_batch - 1

                padding_step_corner[i, 0, mat[0]] -= 1 #- 1
                padding_step_corner[i, 0, mat[1]] -= 1 #- 1
                padding_step_corner[i, 1, mat[0]] -= 1 #- 1
                padding_step_corner[i, 1, mat[1]] += self.N_voxels_in_batch #- 1
                padding_step_corner[i, 2, mat[0]] += self.N_voxels_in_batch #- 1
                padding_step_corner[i, 2, mat[1]] -= 1 #- 1
                padding_step_corner[i, 3, mat[0]] += self.N_voxels_in_batch #- 1
                padding_step_corner[i, 3, mat[1]] += self.N_voxels_in_batch #- 1
                padding_step_corner[i, 4, vec] -= 1 #- 1
                padding_step_corner[i, 5, vec] += self.N_voxels_in_batch #- 1

                padding_step_corner_inner[i, 1, mat[1]] += self.N_voxels_in_batch - 1
                padding_step_corner_inner[i, 2, mat[0]] += self.N_voxels_in_batch - 1
                padding_step_corner_inner[i, 3, mat[0]] += self.N_voxels_in_batch - 1
                padding_step_corner_inner[i, 3, mat[1]] += self.N_voxels_in_batch - 1
                padding_step_corner_inner[i, 5, vec] += self.N_voxels_in_batch - 1

            index = torch.arange(0, self.N_batch_num, device=self.device)
            index = [index, index, index]
            index = torch.stack(torch.meshgrid(index), dim=-1).view(1, -1, 3)
            coord = index * self.voxel_size[:, None, :] * self.N_voxels_in_batch  # * self.N_batch_num # todo todo
            coord = coord + self.voxel_bbox[:, None, 0]

            index_1d = self.index_3d_to_index_1d(index, float(self.N_batch_num)) # todo
            reindex = []
            for j in range(self.N_block_size):
                batch_reindex_j = self.reindex_batch_1d_all[j, index_1d[0].long()]
                reindex.append(batch_reindex_j[None, ...])
            reindex = torch.cat(reindex, dim=0)
            padding_step_edge_inner = padding_step_edge_inner.view(1, 1, 3, 4, -1, 3) / self.N_voxels_in_batch * 2 - 1
            padding_step_edge_inner = padding_step_edge_inner.expand(self.N_block_size, self.N_batch_num**3, 3, 4, self.N_voxels_in_batch, 3)
            reindex_edge = reindex.view(self.N_block_size, self.N_batch_num**3, 1, 1, 1, 1)
            reindex_edge = reindex_edge.expand(self.N_block_size, self.N_batch_num**3, 3, 4, self.N_voxels_in_batch, 1)
            padding_step_corner_inner = padding_step_corner_inner.view(1, 1, 3, 6, 3) / self.N_voxels_in_batch * 2 - 1
            padding_step_corner_inner = padding_step_corner_inner.expand(self.N_block_size, self.N_batch_num**3, 3, 6, 3)
            reindex_corner = reindex.view(self.N_block_size, self.N_batch_num**3, 1, 1, 1)
            reindex_corner = reindex_corner.expand(self.N_block_size, self.N_batch_num**3, 3, 6, 1)
            padding_coord_edge = torch.cat([padding_step_edge_inner, reindex_edge], dim=-1).view(self.N_block_size, -1, 3)
            padding_coord_corner = torch.cat([padding_step_corner_inner, reindex_corner], dim=-1).view(self.N_block_size, -1, 3)
            padding_coord_inner = torch.cat([padding_coord_edge, padding_coord_corner], dim=1)
            padding_coord_inner = padding_coord_inner.view(-1, 4)

            # coord = coord + self.bbox[:, None, 0] + r[:, None, :] * 0.5

            origin = coord

            padding_step_edge = padding_step_edge.view(
                1, 1, 3, 4, self.N_voxels_in_batch, 3
            ) * self.voxel_size[:, None, None, None, None, :]
            padding_step_corner = padding_step_corner.view(1, 1, 3, 6, 3) * self.voxel_size[:, None, None, None, :]

            padding_coord_edge = origin.view(self.N_block_size, -1, 1, 1, 1, 3) + padding_step_edge
            padding_coord_corner = origin.view(self.N_block_size, -1, 1, 1, 3) + padding_step_corner
            T = padding_coord_edge.shape[-2]
            M = padding_coord_edge.shape[1]
            # _ = padding_coord_edge.shape #  (1, 8, 3, 4, self.N_voxels_in_batch, 3)
            # _ = padding_coord_corner#  (1, 8, 3, 5, 3)
            padding_coord_edge = padding_coord_edge.view(self.N_block_size, -1, 3)
            padding_coord_corner = padding_coord_corner.view(self.N_block_size, -1, 3)
            N_e = padding_coord_edge.shape[1]
            N_c = padding_coord_corner.shape[1]
            padding_coord_all = torch.cat([padding_coord_edge, padding_coord_corner], dim=1)
            padding_coord_one = padding_coord_all.view(-1, 3)
            # samples_active, samples_index, sample_valid_ = self._cal_valids(padding_coord_one)
            samples_active, samples_index = self._split_samples_to_batch(padding_coord_one)
            N = samples_active.shape[1]

            index_3d_float = self.get_index_from_points(samples_active, dtype="float")
            # index_3d_float.view(self.N_block_size*M, -1)[self.reindex_batch_1d_global][:, :3*4*T*3].view(1, 56, 3, 4, T, -1)[0,7,0,:,0]
            # index_3d_float[:, :N_e].view(8, 8, 3, 4, T, -1)[0,0,0,0]
            # sample_valid = self.valid_mask(index_3d_float, self.N_voxels_in_batch)
            sample_valid = self.valid_mask(index_3d_float)
            # index_3d_float[:, :N_e, :].view(1, M, 3, 4, T, -1)[0, 0, 0,]
            batch_index = (index_3d_float + 1e-8) / float(self.N_voxels_in_batch) # todo todo
            # sample_valid = batch_index >= 0 and batch_index < self.N_batch_num
            index_inside_batch = batch_index - torch.trunc(batch_index)
            batch_index_1d = self.index_3d_to_index_1d(torch.trunc(batch_index), float(self.N_batch_num))
            batch_index_1d[~sample_valid] = 0
            batch_reindex = []
            for j in range(self.N_block_size):
                batch_reindex_j = self.reindex_batch_1d_all[j, batch_index_1d[j].long()] # todo long
                batch_reindex.append(batch_reindex_j[None, ...])
            batch_reindex = torch.cat(batch_reindex, dim=0).view(self.N_block_size, -1, 1)
            # batch_reindex = self.reindex_batch_1d_all[0, batch_index_1d[0]].view(self.N_block_size, -1, 1)
            batch_reindex[~sample_valid] = 0
            sample_valid = batch_reindex != 0
            # batch_reindex = batch_reindex / self.reindex_batch_1d_all.max() * 2 - 1
            # batch_reindex = batch_reindex / self.reindex_batch_1d_all.max() * 2 - 1

            # index_inside_batch = (index_inside_batch * self.N_voxels_in_batch + 0.5) / self.N_voxels_in_batch
            # index_inside_batch = (index_inside_batch * self.N_voxels_in_batch - 0.5) / (self.N_voxels_in_batch - 1)
            index_sparse = index_inside_batch * 2. - 1.

            index_sparse[~sample_valid.expand(self.N_block_size, N, 3)] = -2.0
            batch_reindex[~sample_valid] = -2.0

            padding_coord_temp = torch.cat([index_sparse, batch_reindex], dim=-1)
            # padding_coord_inner = - 10 * torch.ones_like(padding_coord_one[:, :1]).repeat(1, 4)
            for i in range(self.N_block_size):
                index_valid = samples_index[i] != -1
                index_valid = torch.logical_and(index_valid, sample_valid[i, :, 0])
                valid_index = samples_index[i][index_valid]
                padding_coord_inner[valid_index] = padding_coord_temp[i, index_valid]
            padding_coord_inner = padding_coord_inner.reshape(self.N_block_size, -1, 4)
            # ((padding_coord_all +1)/2.)[((padding_coord_all +1)/2.) >= 1.]

            padding_coord_edge = padding_coord_inner[:, :N_e].view(self.N_block_size, M, 3, 4, T, 4)
            padding_coord_corner = padding_coord_inner[:, -N_c:].view(self.N_block_size, M, 3, 6, 4)

            valid_batch = self.reindex_batch_1d_all.view(-1) != 0
            padding_coord_edge = padding_coord_edge.reshape(1, -1, 3, 4, T, 4)[:, valid_batch]
            padding_coord_corner = padding_coord_corner.reshape(1, -1, 3, 6, 4)[:, valid_batch]
            # padding_coord_all = padding_coord_all.view(self.N_block_size*M, -1)[self.reindex_batch_1d_global]
            # A = padding_coord_all.shape[0]

            self.padding_coord_edge = padding_coord_edge.detach()
            self.padding_coord_corner = padding_coord_corner.detach()
            # (padding_coord_edge[0,6,0,:,0] +1)/2.*32.
            # padding_coord_edge[0, 7, 0, :, 0, :]

    def _set_batched_attr(self):
        pass

    def build(self, points, reindex_start):
        points_index = self.get_index_from_points(points)
        points_index_unique = torch.unique(points_index, dim=1).long()
        index_valid = self.valid_mask(points_index_unique)
        points_index_unique = points_index_unique[index_valid].view(1, -1, 3)

        self._set_active(points_index_unique)
        self._set_active_batch(points_index_unique)
        self._set_reindex_batch(reindex_start)

        if DEBUG_SAVE:
            # self._set_index()
            # self._set_voxel_coord()
            torch.save(points, debug_path + "/input.pt")
            torch.save(self.active, debug_path + "/active.pt")
            # torch.save(self.index, debug_path + "/index.pt")
            # torch.save(self.reindex, debug_path + "/reindex.pt")
            # torch.save(self.voxel_coord, debug_path + "/voxel_coord.pt")
            # torch.save(self.reindex_batch, debug_path + "/reindex_batch.pt")

    def upsample(self, voxel_num):
        # voxel_num = torch.tensor(voxel_num, device=self.device)

        self.N_voxels_in_batch = voxel_num
        self.N_voxels_every_dim = self.N_batch_num * self.N_voxels_in_batch

        # voxel_num = torch.tensor(voxel_num, device=self.device)
        # batch_voxel_res = self.batch_voxel_res / self.voxel_num * voxel_num
        # batch_voxel_res = batch_voxel_res.round().int()
        # self.voxel_num = voxel_num
        # self.batch_voxel_res = batch_voxel_res

        bbox_size = self.bbox[:, 1] - self.bbox[:, 0]
        self.voxel_size_old = self.voxel_size
        self.voxel_size = bbox_size / self.N_voxels_every_dim
        self.voxel_bbox[:, 0] = self.bbox[:, 0] + self.voxel_size / 2
        self.voxel_bbox[:, 1] = self.bbox[:, 1] - self.voxel_size / 2
        # self.bbox[1] = self.bbox[0] + self.voxel_size * (self.voxel_num - 1)

        points_index = self.get_index_from_points(self.points)
        self._set_active(points_index)
        self._set_padding_coord()
        # points_index_unique = torch.unique(points_index, dim=1).long()

        # self._set_active(points_index_unique)
        # self._set_active_batch(points_index_unique)
        # self._set_reindex_batch()

    def update_factor_tensor(self, points_confs_plane, points_confs_line):
        raise NotImplementedError
        L = self.reindex_num[0]
        points_confs_plane = torch.nn.functional.interpolate(
            points_confs_plane,
            size=(L, L),
            mode="bilinear",
            align_corners=True,
            # mode="nearest",
        )
        points_confs_line = torch.nn.functional.interpolate(
            points_confs_line,
            size=(L, 1),
            mode="bilinear",
            align_corners=True
            # mode="nearest",
        )
        return points_confs_plane, points_confs_line

    def valid_mask(self, index, voxel_dim=None):
        if voxel_dim is None:
            voxel_dim = self.N_voxels_every_dim
        valid = torch.logical_and(index >= 0, index < self.N_voxels_every_dim)
        valid = valid[:, :, 0] * valid[:, :, 1] * valid[:, :, 2]
        return valid

    def clip_index(self, index):
        index = torch.clip(index, 0.0, self.N_voxels_every_dim - 1)
        return index

    def _cal_coords(self, samples, sample_valid, matMode, vecMode, detach):
        B, N, _ = samples.shape
        index_3d_float = self.get_index_from_points(samples, dtype="float")
        batch_index = (index_3d_float + 1e-8) / self.N_voxels_in_batch
        index_inside_batch = batch_index - batch_index.long()
        batch_index_1d = self.index_3d_to_index_1d(batch_index.long(), self.N_batch_num)
        batch_index_1d[~sample_valid] = 0
        batch_reindex = [
            self.reindex_batch_1d_all[i, batch_index_1d[i]] for i in range(B)
        ]
        batch_reindex = torch.stack(batch_reindex).view(B, N, 1)
        sample_valid = torch.logical_and(batch_reindex != 0, sample_valid[..., None])

        # batch_reindex[~sample_valid] = 0
        # batch_reindex_invalid = batch_reindex == 0.0

        batch_reindex = batch_reindex / self.reindex_batch_1d_global.max() * 2 - 1
        # batch_reindex = (batch_reindex-0.5) / self.reindex_batch_1d_global.max() * 2 - 1
        # index_inside_batch = (index_inside_batch * self.N_voxels_in_batch + 1) / self.N_voxels_in_batch
        # ]index_inside_batch = (index_inside_batch * self.N_voxels_in_batch - 0.5) / (self.N_voxels_in_batch-1)
        index_sparse = index_inside_batch * 2 - 1
        if torch.sum(~sample_valid) != 0.0:
            for i in range(self.N_block_size):
                index_sparse[i][~sample_valid[i, :].expand(N, 3)] = -2.0
                batch_reindex[i, ~sample_valid[i, :]] = -2.0

        if self.padding_tensor_on:
            # index_sparse = index_sparse
            index_sparse = index_sparse * (
                self.N_voxels_in_batch / (self.N_voxels_in_batch + 2)
            )
            # index_sparse = (index_sparse + 1) / 2
            # index_sparse = index_sparse * self.N_voxels_in_batch + 1
            # index_sparse = index_sparse / (self.N_voxels_in_batch + 2)
            # index_sparse = index_sparse * 2 - 1

        coordinate_plane = torch.stack(
            (
                index_sparse[..., self.matMode[0]],
                index_sparse[..., self.matMode[1]],
                index_sparse[..., self.matMode[2]],
            )
        ).view(3, B, N, 2)
        coordinate_line = torch.stack(
            (
                index_sparse[..., self.vecMode[0]],
                index_sparse[..., self.vecMode[1]],
                index_sparse[..., self.vecMode[2]],
            )
        ).view(3, B, N, 1)

        if detach:
            coordinate_plane = coordinate_plane.detach()
            coordinate_line = coordinate_line.detach()

        batch_reindex = batch_reindex.view(1, B, N, 1).expand(3, B, N, 1)
        # coordinate_plane = torch.cat([batch_reindex, coordinate_plane], dim=-1)
        # coordinate_line = torch.cat([batch_reindex, coordinate_line], dim=-1)
        coordinate_plane = torch.cat([coordinate_plane, batch_reindex], dim=-1)
        coordinate_line = torch.cat([coordinate_line, batch_reindex], dim=-1)



        if DEBUG_SAVE:
            batch_coord = torch.arange(0, self.N_batch_num, device=self.device)
            batch_coord = [batch_coord, batch_coord, batch_coord]
            batch_coord = torch.stack(torch.meshgrid(batch_coord), dim=-1)
            batch_coord = batch_coord.reshape(-1, 3)
            batch_coord = batch_coord * self.voxel_size * self.N_voxels_in_batch
            batch_coord = batch_coord + self.voxel_bbox[0, 0]
            D = self.N_batch_num
            batch_coord = batch_coord.view(D, D, D, 3)
            batch_index = batch_index[0].long()
            batch_points = batch_coord[
                batch_index[:, 0], batch_index[:, 1], batch_index[:, 2]
            ]
            torch.save(batch_points, debug_path + "/batch_points.pt")

        return coordinate_plane, coordinate_line

    def _split_samples_to_batch(self, samples):
        N_max = 0
        sample_list = []
        sample_index = torch.arange(samples.shape[0], device=samples.device)
        sample_index_list = []

        for i in range(self.N_block_size):
            index = self.get_index_from_points(samples[None, ...])[i]
            valid = self.valid_mask(index[None, ...])[0]
            sample_in_bbox = samples[valid]
            index_in_bbox = sample_index[valid]
            sample_list.append(sample_in_bbox)
            sample_index_list.append(index_in_bbox)

            samples = samples[~valid]
            sample_index = sample_index[~valid]
            N_max = max(N_max, sample_in_bbox.shape[0])

        sample_batch = torch.zeros((self.N_block_size, N_max, 3), device=samples.device)
        sample_batch = (
            sample_batch + self.voxel_bbox[:, :1] - self.voxel_size[:, None, :] * 2.0
        )
        index_batch = torch.zeros(
            (self.N_block_size, N_max), device=samples.device, dtype=torch.long
        )

        index_batch = index_batch - 1
        for i in range(self.N_block_size):
            sample_batch[i, : sample_list[i].shape[0]] = sample_list[i]
            index_batch[i, : sample_list[i].shape[0]] = sample_index_list[i].long()
        return sample_batch, index_batch

    def _cal_valids(self, samples):
        # mask out invalid samples
        sample_batch, sample_index = self._split_samples_to_batch(samples)
        index_block = self.get_index_from_points(sample_batch)
        index_block_valid = self.valid_mask(index_block)
        index_block = self.clip_index(index_block)
        index_block_1d = self.index_3d_to_index_1d(index_block, self.N_voxels_every_dim)

        N_max = 0
        sample_list = []
        sample_index_list = []
        # sample_valid_list = []

        for i in range(self.N_block_size):
            index_block_valid_i = index_block_valid[i]
            index_block_active_i = self.active[i][
                index_block_1d[i][index_block_valid_i]
            ]
            # index_block_valid_i[index_block_valid_i] = index_block_active_i
            # index_block_valid_i = torch.logical_and(
            #   index_block_active_i, index_block_valid[i]
            # )
            # index_block_valid_i = index_block_active_i
            sample_list.append(
                sample_batch[i, index_block_valid_i][index_block_active_i]
            )
            sample_index_list.append(
                sample_index[i, index_block_valid_i][index_block_active_i]
            )
            # sample_valid_list.append(index_block_valid_i)
            N_max = max(N_max, sample_list[i].shape[0])

        samples_valid = torch.zeros(
            (self.N_block_size, N_max), device=samples.device, dtype=torch.bool
        )
        # samples_valid = torch.stack(sample_valid_list)
        samples = torch.zeros((self.N_block_size, N_max, 3), device=samples.device)
        samples = samples + self.voxel_bbox[:, :1] - self.voxel_size[:, None, :] * 2.0
        samples_index = torch.zeros(
            (self.N_block_size, N_max), device=samples.device, dtype=torch.long
        )
        samples_index = samples_index - 1
        for i in range(self.N_block_size):
            samples[i, : sample_list[i].shape[0]] = sample_list[i]
            samples_index[i, : sample_list[i].shape[0]] = sample_index_list[i].long()
            samples_valid[i, : sample_list[i].shape[0]] = True
        return samples, samples_index, samples_valid

    def padding_tensor(self, factor_tensor_plane, factor_tensor_line, idx_plane):
        if not self.padding_tensor_on:
            return factor_tensor_plane, factor_tensor_line
        # 1, C, B, H, W
        # 1, C, B, H
        # sampling padding tensor feature
        mat = self.matMode[idx_plane] + [-1]
        vec = [self.vecMode[idx_plane], -1]

        # padding_coord_edge = torch.cat(self.padding_coord_edge, dim=1)
        # padding_coord_corner = torch.cat(self.padding_coord_corner, dim=1)

        padding_coord_edge = self.padding_coord_edge[:, :, idx_plane, :, :, mat]
        padding_coord_corner = self.padding_coord_corner[:, :, idx_plane]
        padding_coord_edge[..., -1] = padding_coord_edge[..., -1] / self.reindex_batch_1d_global.max() * 2 - 1
        padding_coord_corner[..., -1] = padding_coord_corner[..., -1] / self.reindex_batch_1d_global.max() * 2 - 1
        # padding_coord_edge[..., -1] = (padding_coord_edge[..., -1]-0.5) / self.reindex_batch_1d_global.max() * 2 - 1
        # padding_coord_corner[..., -1] = (padding_coord_corner[..., -1]-0.5) / self.reindex_batch_1d_global.max() * 2 - 1
        padding_coord_cor_p = padding_coord_corner[..., mat]
        padding_coord_cor_l = padding_coord_corner[..., vec]


        G = self.reindex_batch_1d_global[-1]
        V = self.N_voxels_in_batch
        C = factor_tensor_plane.shape[1]
        B = self.N_block_size

        # _, N, _, T, _ = padding_coord_edge.shape
        # padding_coord_edge = padding_coord_edge.view(B, -1, 3)
        # N = padding_coord_edge.shape[1]


        # coordinate_plane = index_sparse[..., mat]
        # batch_reindex = batch_reindex.view(1, B, N, 1).expand(3, B, N, 1)
        # coordinate_plane = torch.cat([batch_reindex, coordinate_plane], dim=-1)
        # coordinate_line = torch.cat([batch_reindex, coordinate_line], dim=-1)
        # padding_coord_edge = torch.cat([coordinate_plane, batch_reindex], dim=-1)

        # padding_reindex = torch.arange(0, G, device=self.device)
        # reindex_edge = padding_reindex.view(1, -1, 1, 1, 1).expand(1, G, 4, V, 1)
        # reindex_corner = padding_reindex.view(1, -1, 1, 1).expand(1, G, 5, 1)

        # padding_coord_edge = torch.cat([padding_coord_edge, reindex_edge], dim=-1)
        # padding_coord_cor_p = torch.cat([padding_coord_cor_p, reindex_corner], dim=-1)
        # padding_coord_cor_l = torch.cat([padding_coord_cor_l, reindex_corner], dim=-1)
        padding_edge_plane = torch.nn.functional.grid_sample(
            factor_tensor_plane,
            padding_coord_edge,
            mode="nearest",
            padding_mode="border",
            # align_corners=False,
        )
        # todo check linear interpolation
        padding_corner_plane = torch.nn.functional.grid_sample(
            factor_tensor_plane,
            padding_coord_cor_p[:, :, :4, None, :],
            mode="nearest",
            padding_mode="border",
            # align_corners=False,
        )
        padding_corner_line_s = torch.nn.functional.grid_sample(
            factor_tensor_line,
            padding_coord_cor_l[:, :, 4, None],
            mode="nearest",
            padding_mode="border",
            # align_corners=False,
        )
        padding_corner_line_e = torch.nn.functional.grid_sample(
            factor_tensor_line,
            padding_coord_cor_l[:, :, 5, None],
            mode="nearest",
            padding_mode="border",
        )

        # concat with original feature
        # factor_tensor_plane = torch.cat(
        #     [
        #         padding_edge_plane[:, :, :, 0:1],
        #         factor_tensor_plane,
        #         padding_edge_plane[:, :, :, 1:2],
        #     ],
        #     dim=-2,
        # )
        factor_tensor_plane = torch.cat(
            [
                padding_edge_plane[:, :, :, 0:1].view(1, C, G, V, 1),
                factor_tensor_plane,
                padding_edge_plane[:, :, :, 1:2].view(1, C, G, V, 1),
            ],
            dim=-1,
        )
        padding_edge_plane_l = torch.cat(
            [
                padding_corner_plane[:, :, :, 0:1],
                padding_edge_plane[:, :, :, 2:3],
                padding_corner_plane[:, :, :, 2:3],
            ],
            dim=-1,
        ) #.view(1, C, G, V + 2, 1)
        padding_edge_plane_r = torch.cat(
            [
                padding_corner_plane[:, :, :, 1:2],
                padding_edge_plane[:, :, :, 3:4],
                padding_corner_plane[:, :, :, 3:4],
            ],
            dim=-1,
        ) #.view(1, C, G, V + 2, 1)
        # factor_tensor_plane = torch.cat(
        #     [padding_edge_plane_l, factor_tensor_plane, padding_edge_plane_r], dim=-1
        # )
        # factor_tensor_line = torch.cat(
        #     [padding_corner_line_s, factor_tensor_line, padding_corner_line_e], dim=-1
        # )
        factor_tensor_plane = torch.cat(
            [padding_edge_plane_r, factor_tensor_plane, padding_edge_plane_l], dim=-2
        )
        factor_tensor_line = torch.cat(
            [padding_corner_line_e, factor_tensor_line, padding_corner_line_s], dim=-1
        )

        return factor_tensor_plane, factor_tensor_line
        # self.padding_coord_edge[0,0,0].detach().cpu().numpy()
        # padding_coord_edge[0, 0].detach().cpu().numpy()
        # padding_edge_plane[0, :3, 0].permute(1, 2, 0).detach().cpu().numpy()
        # padding_edge_plane_l[0, :3, 0].permute(1, 2, 0).detach().cpu().numpy()
        # factor_tensor_plane[0, :3, 0, :, :].permute(1, 2, 0).detach().cpu().numpy()
        # factor_tensor_plane[0, 3:6, 0, :, :].permute(1, 2, 0).detach().cpu().numpy() - [[[3,4,5]]]
    #   factor_tensor_line[0, :3].permute(1, 2, 0).detach().cpu().numpy()
    # self.padding_coord_corner[0, :, 2,-2 :, :]

    def query(
        self, samples, factor_tensor_plane, factor_tensor_line, matMode, vecMode, detach
    ):
        # initial features
        samples_active, samples_index, sample_valid = self._cal_valids(samples)

        coordinate_plane, coordinate_line = self._cal_coords(
            samples_active, sample_valid, None, None, detach
        )

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(factor_tensor_line)):
            sample_line, sample_plane = self._interpolate_feature(
                coordinate_plane[idx_plane],
                coordinate_line[idx_plane],
                factor_tensor_plane[idx_plane],
                factor_tensor_line[idx_plane],
                idx_plane,
            )
            plane_coef_point.append(sample_plane)
            line_coef_point.append(sample_line)

        N = samples.shape[0]
        L = len(factor_tensor_plane)
        C = factor_tensor_plane[0].shape[1]
        B, S, _ = samples_active.shape

        plane_coef_point_full = torch.zeros((L * C, N), device=self.device)
        line_coef_point_full = torch.zeros((L * C, N), device=self.device)
        plane_coef_point = torch.cat(plane_coef_point).view(L * C, B, S)
        line_coef_point = torch.cat(line_coef_point).view(L * C, B, S)

        for i in range(self.N_block_size):
            index_valid = sample_valid[i]
            valid_index = samples_index[i][index_valid]
            plane_coef_point_full[:, valid_index] = plane_coef_point[:, i, index_valid]
            line_coef_point_full[:, valid_index] = line_coef_point[:, i, index_valid]

        if DEBUG_SAVE:
            # torch.save(plane_coef_point_full, debug_path + "/plane_coef_point_full.pt")
            # torch.save(line_coef_point_full, debug_path + "/line_coef_point_full.pt")
            torch.save(samples, debug_path + "/samples.pt")
            torch.save(samples_active, debug_path + "/samples_active.pt")
            samples_from_index = samples[samples_index[0]][index_valid]
            torch.save(samples_from_index, debug_path + "/samples_from_index.pt")
            # torch.save(index_block, debug_path + "/index_block.pt")
            # torch.save(index_block_active, debug_path + "/index_block_active.pt")
            # torch.save(index_sparse_near, debug_path + "/index_sparse_near.pt")
            # torch.save(index_block_1d, debug_path + "/index_block_1d.pt")
            # torch.save(weight, debug_path + "/weight.pt")

        return plane_coef_point_full, line_coef_point_full

    def _interpolate_feature(
        self,
        coordinate_plane,
        coordinate_line,
        factor_tensor_plane,
        factor_tensor_line,
        idx_plane,
    ):
        B, N, _ = coordinate_plane.shape
        C = factor_tensor_plane.shape[1]
        coordinate_plane = coordinate_plane.view(1, 1, B * N, 1, 3)
        coordinate_line = coordinate_line.view(1, 1, B * N, 2)

        factor_tensor_plane, factor_tensor_line = self.padding_tensor(
            factor_tensor_plane, factor_tensor_line, idx_plane
        )

        # factor_tensor_plane[0,:3,0].permute(1, 2, 0).detach().cpu().numpy()
        sample_plane = torch.nn.functional.grid_sample(
            factor_tensor_plane,  # 1, C, B, H, W
            coordinate_plane,
            padding_mode="border",
            align_corners=False,
        )
        sample_line = torch.nn.functional.grid_sample(
            factor_tensor_line,  # 1, C, B, H
            coordinate_line,
            padding_mode="border",
            align_corners=False,
        )
        sample_plane = sample_plane.view(C, B, N)
        sample_line = sample_line.view(C, B, N)

        return sample_line, sample_plane
    # factor_tensor_plane[0, :3, :].view(3, 279, 100).permute(1, 2, 0).detach().cpu().numpy()

    def to(self, device):
        for att in self.__dict__:
            if isinstance(self.__dict__[att], torch.Tensor):
                self.__dict__[att] = self.__dict__[att].to(device)
                if device == torch.device("cpu"):
                    self.__dict__[att].detach_()
            elif isinstance(self.__dict__[att], list):
                for i in range(len(self.__dict__[att])):
                    if isinstance(self.__dict__[att][i], torch.Tensor):
                        self.__dict__[att][i] = self.__dict__[att][i].to(device)
                        if device == torch.device("cpu"):
                            self.__dict__[att][i].detach_()
        self.device = device
        return self

    def deepcopy(self):
        block = Block()
        for att in self.__dict__:
            if isinstance(self.__dict__[att], torch.Tensor):
                block.__dict__[att] = self.__dict__[att].clone()
            else:
                block.__dict__[att] = self.__dict__[att]
        return block

    def cat(self, block_list):
        if len(block_list) == 0:
            self.reindex_batch_1d_global = self.reindex_batch_1d
            self._set_padding_coord()
            # self.reindex_batch_1d_all = self.reindex_batch_1d_all.view(1, -1)
            return self

        self.N_block_size = self.N_block_size + len(block_list)
        attr_cat = [
            "active",
            "bbox",
            "voxel_bbox",
            "voxel_size",
            "reindex_batch_1d_all",
            "points",
        ]
        att_list = [
            "active_batch_1d",
            "reindex_batch_1d",
            "padding_coord_edge",
            "padding_coord_corner",
        ]

        B = len(self.reindex_batch_1d)

        for i in range(len(block_list)):
            a = self
            b = block_list[i]
            B += len(b.reindex_batch_1d)

            for att in attr_cat:
                self.__dict__[att] = torch.cat(
                    [a.__dict__[att], b.__dict__[att]], dim=0
                )
            for att in att_list:
                if i == 0:
                    self.__dict__[att] = [a.__dict__[att]]
                self.__dict__[att] += [b.__dict__[att]]

        self.reindex_batch_1d_global = self.reindex_batch_1d
        self.reindex_batch_1d_global = torch.cat(self.reindex_batch_1d_global)
        self._set_padding_coord()
        # if self.padding_tensor_on:
        #     self.padding_coord_edge = torch.cat(self.padding_coord_edge, dim=1)
        #     self.padding_coord_corner = torch.cat(self.padding_coord_corner, dim=1)

        # index = [
        #     torch.arange(
        #         0,
        #         int(self.N_voxels_every_dim / self.N_voxels_in_batch),
        #         device=self.device,
        #     )
        #     for i in range(3)
        # ]
        # index = torch.stack(torch.meshgrid(index), dim=-1).reshape(-1, 3)
        # index = index.view(1, -1, 3).repeat(self.N_block_size, 1, 1)
        # coord = (
        #     index * self.voxel_size[:, None, :] * self.N_voxels_in_batch
        #     + self.voxel_bbox[:, :1]
        # )
        # coord = torch.cat([coord, self.reindex_batch_1d_all[:, :, None]], dim=-1)
        # if DEBUG_SAVE:
        #     torch.save(coord, debug_path + "/reindex_batch_1d_all.pt")

        # self.reindex_batch_1d = torch.arange(0, B) + 1
        # N_batch = int(self.N_batch_num**3)
        # reindex_batch_all = torch.zeros(
        #     (self.N_block_size, N_batch), device=self.device, dtype=torch.long
        # )
        #
        # reindex_end = 0
        # for i in range(self.N_block_size):
        #     # reindex_start = reindex_end
        #     # reindex_end += len(self.reindex_batch_1d_global[i])
        #     reindex = self.reindex_batch_1d[reindex_start:reindex_end]
        #     reindex_batch_all[i][self.active_batch_1d[i]] = reindex
        #
        # self.reindex_batch_1d_all = reindex_batch_all

        return self


#
# DEBUG = False
# if DEBUG:
#     input_points = torch.rand(50, 3)
#     block_voxel = [12, 12, 12]
#     block_test = Block(points=input_points, voxel_num=block_voxel, dilate_kernel=1)
#     # input_points = block_test.voxel_coord.reshape(-1, 3)
#     # block_test = block(points=input_points, voxel_num=block_voxel, dilate_kernel=1)
#
#     import os
#     import open3d as o3d
#
#     # ============ build ============ #
#     # 0. vis input
#     pcd = o3d.geometry.PointCloud()
#     points = input_points.detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = np.zeros((points.shape[0], 3))
#     colors[:, 2] = 1.0
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "input.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 1. vis active
#     pcd = o3d.geometry.PointCloud()
#     points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = np.zeros((points.shape[0], 3))
#     colors[block_test.active.reshape(-1).detach().cpu().numpy(), 0] = 1.0
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "active.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 2. vis index
#     pcd = o3d.geometry.PointCloud()
#     points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = block_test.index.reshape(-1, 3).detach().cpu().numpy() / (
#         block_test.voxel_num - 1
#     )
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "index.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 3. vis reindex
#     pcd = o3d.geometry.PointCloud()
#     points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = block_test.reindex.reshape(-1, 3).detach().cpu().numpy() / (
#         block_test.reindex_num - 1
#     )
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "reindex.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # ============ query ============ #
#
#     def init_kd_svd(n_component, gridSize, scale, device):
#         plane_coef, line_coef = [], []
#         for i in range(3):
#             plane_c, line_c = block_test.init_factor_tensor(n_component[i], device)
#             plane_coef.append(torch.nn.Parameter(scale * plane_c))
#             line_coef.append(torch.nn.Parameter(line_c))
#         # return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
#         return plane_coef.to(device), line_coef.to(device)
#
#     query_points = torch.rand(20, 3)  # * 3.0
#
#     n_layer = 3
#     n_point = query_points.shape[0]
#     n_comp = [1, 1, 1]
#     matMode = [[0, 1], [0, 2], [1, 2]]
#     vecMode = [2, 1, 0]
#     n_layer = 3
#     matMode = [[0, 1], [0, 1], [0, 1]]
#     vecMode = [2, 2, 2]
#     feature_plane, feature_line = init_kd_svd(
#         n_comp, block_test.reindex_num, 0.1, "cpu"
#     )
#
#     # =========== query =============
#     plane_coef_point, line_coef_point = block_test.query(
#         query_points, feature_plane, feature_line, matMode, vecMode, True
#     )
#     # query_feature = (plane_coef_point * line_coef_point).T
#     #
#     # plane_coef_point, line_coef_point = block_test.query(
#     #     block_test.voxel_coord.reshape(-1, 3),
#     #     feature_plane,
#     #     feature_line,
#     #     matMode,
#     #     vecMode,
#     #     True,
#     # )
#     # voxel_feature = (plane_coef_point * line_coef_point).T
#     # ================================
#
#     # =========== update =============
#     res_target = [6, 6, 6]
#     block_test.update(res_target)
#     for i in range(3):
#         coef = block_test.update_factor_tensor(
#             feature_plane[i].data, feature_line[i].data
#         )
#         feature_plane[i] = torch.nn.Parameter(coef[0])
#         feature_line[i] = torch.nn.Parameter(coef[1])
#     plane_coef_point_ups, line_coef_point_ups = block_test.query(
#         query_points, feature_plane, feature_line, matMode, vecMode, True
#     )
#
#     # 1. vis active
#     pcd = o3d.geometry.PointCloud()
#     points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = np.zeros((points.shape[0], 3))
#     colors[block_test.active.reshape(-1).detach().cpu().numpy(), 0] = 1.0
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "active_ups.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 2. vis index
#     pcd = o3d.geometry.PointCloud()
#     points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = block_test.index.reshape(-1, 3).detach().cpu().numpy() / (
#         block_test.voxel_num - 1
#     )
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "index_ups.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 3. vis reindex
#     pcd = o3d.geometry.PointCloud()
#     points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = block_test.reindex.reshape(-1, 3).detach().cpu().numpy() / (
#         block_test.reindex_num - 1
#     )
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "reindex_ups.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # ================================
#
#     # plane_coef_point = plane_coef_point.reshape(n_layer, -1, n_point)
#     # line_coef_point = line_coef_point.reshape(n_layer, -1, n_point)
#     # sigma_feature = torch.zeros((n_point,), device=query_points.device)
#     # for idx_plane in range(n_layer):
#     #     sigma_feature = sigma_feature + torch.sum(plane_coef_point[idx_plane] * line_coef_point[idx_plane], dim=0)
#
#     # 0. vis query points
#     pcd = o3d.geometry.PointCloud()
#     points = query_points.detach().cpu().numpy()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     colors = np.zeros((points.shape[0], 3))
#     colors[:, 2] = 1.0
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     pcd_filepath = os.path.join(debug_path, "query.ply")
#     o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 1. vis query feature
#     # pcd = o3d.geometry.PointCloud()
#     # points = query_points.detach().cpu().numpy()
#     # pcd.points = o3d.utility.Vector3dVector(points)
#     # colors = query_feature.detach().cpu().numpy()
#     # colors /= colors.max()
#     # pcd.colors = o3d.utility.Vector3dVector(colors)
#     # pcd_filepath = os.path.join(debug_path, "query_feat.ply")
#     # o3d.io.write_point_cloud(pcd_filepath, pcd)
#     #
#     # # 2. vis voxel feature
#     # pcd = o3d.geometry.PointCloud()
#     # points = block_test.voxel_coord.reshape(-1, 3).detach().cpu().numpy()
#     # pcd.points = o3d.utility.Vector3dVector(points)
#     # colors = voxel_feature.detach().cpu().numpy()
#     # colors /= colors.max()
#     # pcd.colors = o3d.utility.Vector3dVector(colors)
#     # pcd_filepath = os.path.join(debug_path, "voxel_feat.ply")
#     # o3d.io.write_point_cloud(pcd_filepath, pcd)
#
#     # 3. vis coef
