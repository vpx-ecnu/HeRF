import torch

from opt import config_parser
from utils import timer
from extra.block import Block
from dataLoader.ray_utils import get_rays, ndc_rays_blender, get_ray_directions
from models.tensoRF import TensorVMSplit
from queue import PriorityQueue
# from memory_profiler import profile


# NOTE: desperate
def block_in_pyramid(pyramid_rays, block_points):
    # TODO: add batch dim for block_points
    N = block_points.shape[0]
    pyramid_rays = pyramid_rays.expand(N, 8, -1, -1)
    block_points = block_points.unsqueeze(dim=-2)  # (N, 8, 1, 3)
    # Extract ray start points and directions
    ray_starts = pyramid_rays[..., :3]  # Shape: (N, 8, 4, 3)
    ray_dirs = pyramid_rays[..., 3:]  # Shape: (N, 8, 4, 3)

    # Compute vectors from the start points to the point of interest
    vectors = block_points - ray_starts  # Shape: (N, 8, 4, 3)
    # Compute dot products between the vectors and ray directions
    dot_products = torch.sum(vectors * ray_dirs, dim=-1)  # Shape: (N, 8, 4)
    # Check if the point lies within the pyramid
    in_pyramid = torch.all(dot_products >= 0, dim=-1).any(dim=-1)  # (N, )
    return in_pyramid


def compute_aabb(points: torch.Tensor):
    """
    Args:
        points: torch.Tensor, [N, 8, 3]
    Returns:
        aabb: torch.Tensor, [N, 2, 3]
    """
    min_vals, _ = torch.min(points, dim=1)
    max_vals, _ = torch.max(points, dim=1)
    min_corner = min_vals
    max_corner = max_vals
    return torch.stack([min_corner, max_corner], dim=1)


@timer
def check_collision(ray_bbox: torch.Tensor, block_bbox: torch.Tensor):
    """
    Args:
        ray_bbox: torch.Tensor, [N_ray, N_max_on_ray, 2, 3]
        block_bbox: torch.Tensor, [N_block, 2, 3]
    Returns:
        collision: torch.Tensor, [N_block, ], boolean
    """
    N_ray, N_max_on_ray = ray_bbox.shape[:2]
    block_bbox = block_bbox.unsqueeze(1).unsqueeze(1)
    block_bbox = block_bbox.expand(
        -1, N_ray, N_max_on_ray, -1, -1
    )  # (N_block, N_ray, N_max_on_ray, 2, 3)
    cube1_min = ray_bbox[..., 0, :]
    cube1_max = ray_bbox[..., 1, :]
    cube2_min = block_bbox[..., 0, :]
    cube2_max = block_bbox[..., 1, :]

    collision = torch.all(cube1_max >= cube2_min, dim=-1) & torch.all(
        cube1_min <= cube2_max, dim=-1
    )
    block_collsion = collision.any(dim=-1).any(dim=-1)
    return block_collsion


class SScene:
    def __init__(self, dataset, bbox_edge_scale=0.01):
        # self.block_list = block_all

        # bbox_list = []
        # for block in self.block_list:
        #     bbox_list.append(block.bbox)
        # self.bbox_all = torch.stack(bbox_list)  # (N_block, 2, 3)
        self.rays_bbox = None
        self.block_vertices_all = None
        self.bbox_all = None
        self.block_list: list[Block] = None
        self.dataset = dataset
        self.poses: torch.Tensor = dataset.poses  # (N, 4, 4)
        self.scene_bbox: torch.Tensor = None
        self.bbox_edge_scale = bbox_edge_scale
        # self.intrinsics: torch.Tensor = dataset.intrinsics  # (3, 3)
        self.near_far: list[int] = dataset.near_far
        self.intrinsics_dir: torch.Tensor = dataset.directions  # (H, W, 3)
        self.intrinsics_dir = self.intrinsics_dir / torch.norm(
            self.intrinsics_dir, dim=-1, keepdim=True
        )
        # self.N_block = len(block_all)
        self.N_pose = self.poses.shape[0]

    def _split_scene(self, N_split):
        """
        Divide the scene equally.
        Args:
            N_split: int, split into N_split part along one axis.
        Returns:
            torch.Tensor, (N_split^3, 2, 3), bboxes of split blocks
        """
        bbox = self.scene_bbox
        x = torch.linspace(bbox[0][0], bbox[1][0], N_split + 1)
        y = torch.linspace(bbox[0][1], bbox[1][1], N_split + 1)
        z = torch.linspace(bbox[0][2], bbox[1][2], N_split + 1)

        xx, yy, zz = torch.meshgrid(x[:-1], y[:-1], z[:-1])
        xx2, yy2, zz2 = torch.meshgrid(x[1:], y[1:], z[1:])

        bboxes = torch.stack(
            [
                xx.flatten(),
                yy.flatten(),
                zz.flatten(),
                xx2.flatten(),
                yy2.flatten(),
                zz2.flatten(),
            ],
            dim=1,
        ).view(-1, 2, 3)
        return bboxes

    def _split_points_kdtree(self, pointcloud, N_split_per_axis):
        N_split = N_split_per_axis**3
        queue = PriorityQueue()
        point_list = [pointcloud]
        queue.put((-len(pointcloud), len(point_list) - 1))
        while queue.qsize() < N_split:
            n_points, idx = queue.get()
            points = point_list[idx].view(-1, 3)
            min_vals = torch.min(points, dim=0)[0]
            max_vals = torch.max(points, dim=0)[0]
            gaps = max_vals - min_vals
            axis = torch.argmax(gaps)
            split_value = torch.median(points[:, axis])
            left_points = points[points[:, axis] <= split_value]
            right_points = points[points[:, axis] > split_value]
            point_list.append(left_points)
            queue.put((-len(left_points), len(point_list) - 1))
            point_list.append(right_points)
            queue.put((-len(right_points), len(point_list) - 1))
        return [point_list[queue.get()[1]] for _ in range(N_split)]

    def _split_pointd_voxel(self, points, N_split_per_axis):
        bboxes = self._split_scene(N_split_per_axis)
        points = points.cpu()  # todo debug
        points_per_bbox = []
        for bbox in bboxes:
            xmin, ymin, zmin = bbox[0]
            xmax, ymax, zmax = bbox[1]
            indices = (
                (points[:, 0] > xmin)
                & (points[:, 0] <= xmax)
                & (points[:, 1] > ymin)
                & (points[:, 1] <= ymax)
                & (points[:, 2] > zmin)
                & (points[:, 2] <= zmax)
            )
            points_per_bbox.append(points[indices])
        return points_per_bbox

    # @staticmethod
    def _split_points(self, points, n_split_per_axis, mode="kdtree"):
        """
        Split point cloud by bboxes
        Args:
            points: torch.Tensor, (N_point, 3)
            bboxes: torch.Tensor, (N_bbox, 2, 3)
        Returns:
            list[torch.Tensor], points_per_bbox
        """
        assert mode in ["kdtree", "voxel"]
        if mode == "voxel":
            points_per_bbox = self._split_pointd_voxel(points, n_split_per_axis)
        elif mode == "kdtree":
            points_per_bbox = self._split_points_kdtree(points, n_split_per_axis)

        return points_per_bbox

    def build_blocks(self, N_bbox_on_ray, n_split_per_axis=10):
        """
        Build block_list in scene and rays_bbox.
        Args:
            N_bbox_on_ray:
            N_split_per_axis:
        """
        # Block(points=points, voxel_num=args.voxel_num, k_query=args.k_query)
        points = self.dataset.points_xyz_all
        voxel_num = self.dataset.opt.voxel_num
        batch_voxel_res = self.dataset.opt.batch_voxel_res

        points_per_bbox = self._split_points(points, n_split_per_axis, mode=self.dataset.opt.partition_mode)
        del points
        # debug_path = "/media/data/yxy/ed-nerf/logs/tensoir/log_original_blender/debug"
        # torch.save(points_per_bbox, debug_path+"/points_per_bbox.pt")

        block_list = []
        reindex_current = 1
        for i in range(len(points_per_bbox)):
            if points_per_bbox[i].shape[0] >= 2:
                block = Block(
                    points=points_per_bbox[i],
                    N_voxels_every_dim=voxel_num[0],
                    N_voxels_in_batch=batch_voxel_res,
                    reindex_start=reindex_current,
                    bbox_edge_scale=0.0,
                    n_block_per_axis=n_split_per_axis,
                    padding_tensor_on=self.dataset.opt.padding_tensor_on,
                    sparse_voxel_on=self.dataset.opt.sparse_voxel_on,
                )
                reindex_current = block.reindex_batch_1d[-1] + 1
                block_list.append(block)

        self.block_list = block_list
        print("block num: ", len(block_list), "/", len(points_per_bbox))
        print("total active batch: ", reindex_current - 1)

        bbox_list = []
        for block in self.block_list:
            bbox_list.append(block.bbox)
        self.bbox_all = torch.cat(bbox_list, dim=0).cpu()  # (N_block, 2, 3)
        # self.pyramid_rays_all: torch.Tensor = self._get_pyramid_rays(dataset.directions)  # Shape: (N_pose, 4, 3)
        self.block_vertices_all: torch.Tensor = self._get_all_blk_vertices()
        # (N_block, 8, 3)
        self.rays_bbox: torch.Tensor = self._get_rays_bbox(N_bbox_on_ray)
        # (N_pose, N_bbox_on_ray, 2, 3)

    def init_kd_svd(self, n_batch, n_voxel, n_component, scale, device):
        raise NotImplementedError
        plane_coef, line_coef = [], []
        for i in range(3):
            B = int(n_batch)
            V = int(n_voxel)
            C = int(n_component[i])
            # line_c = torch.arange(1, C * B * V + 1, device=device, dtype=torch.float) #/ (B * V)
            # plane_c = torch.arange(1, C * B * V * V + 1, device=device, dtype=torch.float) #/ (B * V * V)
            # line_c = line_c.reshape(1, C, B, V)
            # plane_c = plane_c.reshape(1, C, B, V, V)
            plane_c = torch.randn([1, C, B, V, V], device=device)
            line_c = torch.randn([1, C, B, V], device=device)
            plane_coef.append(torch.nn.Parameter(scale * plane_c))
            line_coef.append(torch.nn.Parameter(scale * line_c))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def build_tensors(self, args, density_n_comp, app_n_comp):
        raise NotImplementedError
        N_batch = self.block_list[-1].reindex_batch_1d[-1]
        N_voxel = self.block_list[0].N_voxels_in_batch
        print("N_batch: ", N_batch)
        assert args.model_name == "TensorVMSplit"
        self.tensor_list = {}

        param_list = ["app", "density"]
        for param in param_list:
            if "app" in param:
                N_component = app_n_comp
                scale = 2.0
            elif "density" in param:
                N_component = density_n_comp
                scale = 1.0
            else:
                raise NotImplementedError

            tensor = self.init_kd_svd(N_batch, N_voxel, N_component, scale, "cpu")
            self.tensor_list.update(
                {f"{param}_plane": tensor[0], f"{param}_line": tensor[1]}
            )

    def init_scene_bbox(self):
        points = self.dataset.points_xyz_all
        point_min = points.min(dim=0)[0]
        point_max = points.max(dim=0)[0]
        bbox_size = point_max - point_min
        point_min -= bbox_size * self.bbox_edge_scale
        point_max += bbox_size * self.bbox_edge_scale
        bbox_size = point_max - point_min
        voxel_num = torch.tensor(self.dataset.opt.voxel_num, device=bbox_size.device)
        voxel_size = bbox_size / (voxel_num - 1)
        point_max = point_min + voxel_size * (voxel_num - 1)
        self.scene_bbox = torch.stack([point_min, point_max])

    @timer
    def get_intersect_blocks_mask(self, pose_idxes: list[int]):
        collision = check_collision(self.rays_bbox[pose_idxes], self.bbox_all)
        active_idx = torch.nonzero(collision).view(-1)  # .tolist()
        return active_idx

    def _get_rays_bbox(self, N_bbox_on_ray):
        rays_o, rays_d = self._get_pyramid_rays()  # (N_pose, 4, 3)
        near, far = self.near_far

        # get vertexes of prism
        z_steps = torch.linspace(0, 1, N_bbox_on_ray + 1, device=rays_o.device)
        z_vals = near * (1 - z_steps) + far * z_steps
        vertexes = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
            -1
        ).unsqueeze(-1)
        # (N_pose, N_bbox_on_ray + 1, 4, 3)

        v_st = vertexes[:, :-1, ...]
        v_ed = vertexes[:, 1:, ...]
        vertexes_8 = torch.cat([v_st, v_ed], dim=-2)  # (N_pose, N_bbox_on_ray, 8, 3)
        aabb_flatten = compute_aabb(vertexes_8.view(-1, 8, 3))
        aabb = aabb_flatten.view(self.N_pose, N_bbox_on_ray, 2, 3)
        return aabb

    def _get_pyramid_rays(self):
        """
        Returns:
            rays_o: tensor.Torch, [N_pose, 4, 3]
            rays_d: tensor.Torch, [N_pose, 4, 3], normalized
        """
        dir_corner = self.intrinsics_dir[[0, 0, -1, -1], [0, -1, -1, 0]]  # (2, 2, 3)
        rays_o_list, rays_d_list = [], []
        for i in range(self.N_pose):
            r_o, r_d = get_rays(dir_corner, self.poses[i])
            rays_o_list.append(r_o)
            rays_d_list.append(r_d)
        rays_o = torch.stack(rays_o_list)  # (N_pose, 4, 3)
        rays_d = torch.stack(rays_d_list)  # (N_pose, 4, 3)
        return rays_o, rays_d

    def _get_all_blk_vertices(self):
        min_coords = self.bbox_all[:, 0, :]  # Shape: (batch_size, 3)
        max_coords = self.bbox_all[:, 1, :]  # Shape: (batch_size, 3)

        combinations = torch.tensor(
            [
                [0, 0, 0],  # Combinations for x, y, and z
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=self.bbox_all.dtype,
            device=self.bbox_all.device,
        )  # Shape: (8, 3)

        vertices = min_coords.unsqueeze(1) + (max_coords - min_coords).unsqueeze(
            1
        ) * combinations.unsqueeze(0)
        # (N_block, 8, 3)

        return vertices

    # @profile
    def split_dataset(self, dataset, camera_group_num):
        N_camera_per_group = int(len(dataset.poses) // (camera_group_num))
        n_dataset = len(dataset.poses) - 1
        i = i_min = i_max = 0

        self.train_split_dataset = []
        while i_min < n_dataset and i_max < n_dataset:
            i_min = i * N_camera_per_group
            i_max = (i + 1) * N_camera_per_group
            i_max = i_max if i_max < n_dataset else n_dataset
            i += 1
            # Todo: maybe generate index list by distance between camera pose
            index_list = torch.arange(i_min, i_max, dtype=torch.long)
            rays, rgbs = dataset.split_by_index(index_list)

            block_id_list = self.get_intersect_blocks_mask(index_list)
            block_list = [self.block_list[i] for i in block_id_list]

            group = (rays, rgbs, block_list, None, block_id_list)
            self.train_split_dataset.append(group)

        del dataset

    def assign_block(self, tensorf, device="cpu"):
        raise NotImplementedError
        tensor_param = ["app_line", "app_plane", "density_line", "density_plane"]
        for param in tensor_param:
            tensor = tensorf.__dict__[param]
            index = tensorf.block.reindex_batch_1d_global - 1
            # index = torch.cat([torch.zeros_like(index[:1]), index], dim=0)

            for i in range(3):
                t = torch.tensor(self.tensor_list[param][i], device=device)
                t[:, :, index] = tensor[i].to(device)
                self.tensor_list[param][i] = t
            # self.tensor_list[param][index] = tensor.to(device)
            # del tensorf.__dict__[param]

        del tensorf.block


if __name__ == "__main__":
    from dataLoader import BlenderDataset

    args = config_parser()
    dataset = BlenderDataset(
        datadir="/media/data/dblu/datasets/nerf_synthetic/lego",
        points_generation_mode="mvsnet",
        args=args,
    )
    sscene = dataset.sscene

    query = list(range(20))
    sscene.get_intersect_blocks(query)
    pass
