import os
import time
import math
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

import models.pointnerf.gen_points as mvs_utils
from dataLoader.ray_utils import *
from extra.block import Block
from extra.sparse_scene import SScene
from utils import timer

from data_preprocess.complete_pcd import *

from dataLoader.ray_utils import *

class BlenderDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=False,
        N_vis=-1,
        points_generation_mode="none",
        block=None,
        args=None,
        **kwargs,
    ):
        self.opt = args
        self.block = block
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.img_hw = (int(800 / downsample), int(800 / downsample))
        self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.points_generation_mode = points_generation_mode
        self.define_transforms()

        # scene_bbox will be modified by point cloud bbox in read_points()
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.0, 8.0]
        self.near_far_point = [0.0, 8.0]

        self.read_points(points_generation_mode, args)

        # self.scene_bbox = self.block.bbox.clone()
        # block_size = self.scene_bbox[1] - self.scene_bbox[0]
        # self.scene_bbox[1] += block_size * 0.2
        # self.scene_bbox[0] -= block_size * 0.2

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):
        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as f:
            self.meta = json.load(f)
        with open(os.path.join(self.root_dir, f"transforms_test.json"), "r") as f:
            self.meta_test = json.load(f)

        if self.opt.debug_load_n_dataset > 0:
            self.meta["frames"] = self.meta["frames"][: self.opt.debug_load_n_dataset]
            self.meta_test["frames"] = self.meta_test["frames"][: self.opt.debug_load_n_dataset]

        h, w = self.img_hw
        # original focal length, fov -> focal
        self.focal = 0.5 * 800.0 / np.tan(0.5 * self.meta["camera_angle_x"])
        # modify focal length to match size self.img_hw
        self.focal *= self.img_hw[0] / 800.0

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            h, w, [self.focal, self.focal]
        )  # (h, w, 3)
        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True
        )
        self.intrinsics = torch.tensor(
            [[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]
        ).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        # self.downsample = 1.0

        if self.points_generation_mode == "mvsnet":
            self.alphas = []
            self.depths = []
            self.mvsimgs = []
            self.render_gtimgs = []
            self.world2cams = []
            self.cam2worlds = []
            self.cam2worlds_test = []
            self.proj_mats = []

        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))

        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            frame = self.meta["frames"][i]
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_hw, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)

            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img_mask = ~(img[:, -1:] == 0)
            self.all_masks += [img_mask.squeeze(0)]
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 6)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_masks = torch.cat(
                self.all_masks, 0
            )  # (len(self.meta['frames])*h*w, 1)
            #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_light_idx = torch.zeros(
                (*self.all_rays.shape[:-1], 1), dtype=torch.long
            )
        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 6)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_hw[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(
                -1, *self.img_hw[::-1]
            )  # (len(self.meta['frames]),h,w,1)
            self.all_light_idx = torch.zeros(
                (*self.all_rays.shape[:-1], 1), dtype=torch.long
            ).reshape(-1, *self.img_hw[::-1])

    def build_view_metas(self):
        self.view_id_list = []
        cam_xyz_lst = [c2w[:3, 3] for c2w in self.cam2worlds]
        test_cam_xyz_lst = [c2w[:3, 3] for c2w in self.cam2worlds_test]

        if self.split == "train":
            cam_xyz = np.stack(cam_xyz_lst, axis=0)
            test_cam_xyz = np.stack(test_cam_xyz_lst, axis=0)
            triangles = mvs_utils.triangluation_bpa(
                cam_xyz, test_pnts=test_cam_xyz, full_comb=self.opt.full_comb > 0
            )
            self.view_id_list = [triangles[i] for i in range(len(triangles))]
            if self.opt.full_comb < 0:
                with open(
                    f"../data/nerf_synth_configs/list/lego360_init_pairs.txt"
                ) as f:
                    for line in f:
                        str_lst = line.rstrip().split(",")
                        src_views = [int(x) for x in str_lst]
                        self.view_id_list.append(src_views)

    @timer
    def generate_points_by_mvsnet(self, args):
        self.build_view_metas()
        (
            points_xyz_all,
            points_embedding_all,
            points_color_all,
            points_dir_all,
            points_conf_all,
            img_lst,
            c2ws_lst,
            w2cs_lst,
            intrinsics_all,
            HDWD_lst,
        ) = mvs_utils.gen_points_filter_embeddings(self, args)
        print(f"mvs point shape: {points_xyz_all.shape[0]}")
        return points_xyz_all, points_conf_all

    # @timer
    # def build_sscene(self, args):
    #     self.sscene = SScene(self)
    #     self.sscene.init_scene_bbox()
    #     self.sscene.build_blocks(args.block_num_on_ray)

    def read_points(self, points_generation_mode, args):
        if points_generation_mode == "mvsnet" and self.split == "train":
            points_xyz_all, points_conf_all = self.generate_points_by_mvsnet(args)
            self.points_xyz_all = points_xyz_all  # will be deleted after build_sscene()
            # self.build_sscene(args)
            # del self.points_xyz_all
            # self.scene_bbox = self.sscene.scene_bbox
            return

        if self.split != "train":
            return
        points_path = "pcd.ply"
        points_path = os.path.join(self.root_dir, points_path)
        points_xyz = read_points_from_path(points_path)

        self.points_xyz_all = points_xyz
        self.init_bbox(points_xyz)
        print(f"points_xyz.shape : {points_xyz.shape}")

    def init_bbox(self, points, scale_thre=0.01):
        points = points
        point_min = points.min(dim=0)[0]
        point_max = points.max(dim=0)[0]
        bbox_size = point_max - point_min
        # TODO: scale
        point_min -= bbox_size * scale_thre
        point_max += bbox_size * scale_thre
        bbox_size = point_max - point_min
        voxel_num = torch.tensor(self.opt.voxel_num, device=bbox_size.device)
        voxel_size = bbox_size / (voxel_num - 1)
        point_min = point_min - voxel_size
        point_max = point_max + voxel_size
        self.scene_bbox = torch.stack([point_min, point_max])
        # self.near_far = [bbox_size.max().item() * 0.1, bbox_size.max().item() * 0.9]
        self.near_far = [0., 8.]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def split_by_index(self, index_list):
        index_wh = torch.arange(0, self.img_hw[0] * self.img_hw[1])
        rays_index = (index_list[:, None] + 1) * index_wh[None, :]
        rays_index = rays_index.reshape(-1)  # todo view and index
        return self.all_rays[rays_index], self.all_rgbs[rays_index]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}

        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]  # for quantity evaluation
            light_idx = self.all_light_idx[idx]
            sample = {
                "img_hw": self.img_hw,  # (int, int)
                "light_idx": light_idx.view(1, -1, 1),
                "rays": rays,
                "rgbs": img.view(1, -1, 3),
                "rgbs_mask": mask,
            }
        return sample

    def get_init_item(self, idx):
        sample = {}
        init_view_num = self.opt.init_view_num
        view_ids = self.view_id_list[idx]
        if self.split == "train":
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        mvs_images, imgs, depths_h, alphas = [], [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = (
            [],
            [],
            [],
            [],
            [],
        )  # record proj mats between views
        for i in view_ids:
            # vid = self.view_id_dict[i]
            vid = i
            # mvs_images += [self.normalize_rgb(self.mvsimgs[vid])]
            # mvs_images += [self.render_gtimgs[vid]]
            mvs_images += [self.mvsimgs[vid]]
            imgs += [self.render_gtimgs[vid]]
            proj_mat_ls = self.proj_mats[vid]
            intrinsics.append(self.intrinsics)
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            depths_h.append(self.depths[vid])
            alphas.append(self.alphas[vid])
            near_fars.append(self.near_far_point)

        for i in range(len(affine_mat)):
            view_proj_mats = []
            ref_proj_inv = affine_mat_inv[i]
            for j in range(len(affine_mat)):
                if i == j:  # reference view
                    view_proj_mats += [np.eye(4)]
                else:
                    view_proj_mats += [affine_mat[j] @ ref_proj_inv]
            # view_proj_mats: 4, 4, 4
            view_proj_mats = np.stack(view_proj_mats)
            proj_mats.append(view_proj_mats[:, :3])
        # (4, 4, 3, 4)
        proj_mats = np.stack(proj_mats)
        imgs = np.stack(imgs).astype(np.float32)
        mvs_images = np.stack(mvs_images).astype(np.float32)

        depths_h = np.stack(depths_h)
        alphas = np.stack(alphas)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = (
            np.stack(intrinsics),
            np.stack(w2cs),
            np.stack(c2ws),
            np.stack(near_fars),
        )

        sample["images"] = imgs  # (V, 3, H, W)
        sample["mvs_images"] = mvs_images  # (V, 3, H, W)
        sample["depths_h"] = depths_h.astype(np.float32)  # (V, H, W)
        sample["alphas"] = alphas.astype(np.float32)  # (V, H, W)
        sample["w2cs"] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample["c2ws"] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample["near_fars_depth"] = near_fars.astype(np.float32)[0]
        sample["near_fars"] = near_fars.astype(np.float32)
        sample["proj_mats"] = proj_mats.astype(np.float32)
        sample["intrinsics"] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample["view_ids"] = np.array(view_ids)
        # sample['light_id'] = np.array(light_idx)
        sample["affine_mat"] = affine_mat
        sample["affine_mat_inv"] = affine_mat_inv

        for key, value in sample.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                    sample[key] = value.unsqueeze(0)
        return sample


if __name__ == "__main__":
    dataset = BlenderDataset(datadir="/media/data/dblu/datasets/nerf_synthetic/lego")
    item = dataset.__getitem__(0)
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"key:{key} tensor.shape:{value.shape}")
        else:
            print(f"key:{key} value:{value.shape}")

    print(f"rays.shape {dataset.all_rays.shape}")  # [640000, 6]

    print(f"rgbs.shape : {dataset.all_rgbs.shape}")  # [640000, 3]
