import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
from plyfile import PlyData, PlyElement
#from memory_profiler import profile

from data_preprocess.complete_pcd import *

from dataLoader.ray_utils import *


class ScannetDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=False,
        N_vis=-1,
        render_train=False,
        args=None,
        **kwargs,
    ):
        self.opt = args
        self.N_vis = N_vis
        self.scan = os.path.basename(datadir)
        self.root_dir = os.path.dirname(datadir)
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()
        # downsample first, and then crop margin
        self.margin = args.scannet_margin
        self.render_train = render_train

        # will be changed in load_point()
        bbox_min = -20.0  # -2.
        bbox_max = 10.0  # 8.
        self.scene_bbox = torch.tensor(
            [[bbox_min, bbox_min, bbox_min], [bbox_max, bbox_max, bbox_max]]
        )
        self.near_far = [0., 10.]
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self.cloudcompare2opencv = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.read_meta()
        self.read_points()
        self.define_proj_mat()
        self.white_bg = False
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        return depth

    def read_meta(self):
        # intrinsics
        self.intrinsics = np.loadtxt(
            os.path.join(
                self.root_dir, self.scan, "intrinsic/intrinsic_color.txt"
            )
        ).astype(np.float32)[:3, :3]
        self.intrinsics = torch.tensor(self.intrinsics).float()
        self.intrinsics[0, :] /= self.downsample
        self.intrinsics[1, :] /= self.downsample
        self.focal_xy = (self.intrinsics[0, 0], self.intrinsics[1, 1])
        # self.depth_intrinsic = np.loadtxt(
        #     os.path.join(self.root_dir, self.scan, "intrinsic/intrinsic_depth.txt")).astype(np.float32)[:3, :3]

        self.build_init_metas()
        w_ori, h_ori = Image.open(self.image_paths[0]).size
        self.img_hw = int(h_ori / self.downsample - 2 * self.margin), int(
            w_ori / self.downsample - 2 * self.margin
        )
        h, w = self.img_hw
        self.img_wh = w, h

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_dir_scannet(
            h_ori / self.downsample,
            w_ori / self.downsample,
            self.intrinsics,
            self.margin,
        )  # (h, w, 3)

        image_paths_split = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = (
            []
        )  # mask of rgba img, here we use rgb img ,so mask should be always valid

        if self.opt.demo_mode:
            print("=============== demo mode ===============")
            self.demo_T = np.loadtxt(
                os.path.join(self.root_dir, self.scan, "demo/demo_trans.txt")).astype(np.float32)

        if self.split == "train":
            idxs = self.train_id_list
            idxs = idxs[::28] if self.opt.debug_mode_smaller_dataset else idxs
        elif self.split == "test":
            idxs = self.test_id_list
            idxs = idxs[::19] if self.opt.debug_mode_smaller_dataset else idxs
        # load data for each img
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            # load pose
            pose = np.loadtxt(
                os.path.join(
                    self.root_dir, self.scan, "pose", "{}.txt".format(i)
                )
            ).astype(np.float32)
            # pose = self.demo_T @ pose @ np.linalg.inv(self.demo_T) if self.opt.demo_mode else pose
            pose = self.demo_T @ pose if self.opt.demo_mode else pose
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            # generate rays
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            del rays_o, rays_d

            # load img path
            # self.image_paths is not split into train and test now
            image_path = self.image_paths[i]
            image_paths_split += [image_path]
            img = Image.open(image_path)

            # downsample
            if self.downsample != 1.0:
                img = img.resize(
                    (
                        self.img_hw[1] + 2 * self.margin,
                        self.img_hw[0] + 2 * self.margin,
                    ),
                    Image.LANCZOS,
                )
            w_down, h_down = img.size
            img = img.crop(
                (self.margin, self.margin, w_down - self.margin, h_down - self.margin)
            )
            img_tensor = self.transform(img)  # (3, h, w)
            del img
            img_tensor = img_tensor.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img_tensor]
            # always valid
            img_mask = torch.ones((h * w, 1), dtype=bool)
            self.all_masks += [img_mask]

        self.image_paths = image_paths_split
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

    def build_init_metas(self):
        colordir = os.path.join(self.root_dir, self.scan, "color")
        self.image_paths = [
            f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))
        ]
        self.image_paths = [
            os.path.join(self.root_dir, self.scan, "color/{}.jpg".format(i))
            for i in range(len(self.image_paths))
        ]
        self.all_id_list = self.filter_valid_id(list(range(len(self.image_paths))))
        if self.opt.load_train_frames[0] != -1:
            self.all_id_list = [t_id for t_id in self.all_id_list if t_id >= self.opt.load_train_frames[0]]
            self.all_id_list = [t_id for t_id in self.all_id_list if t_id <= self.opt.load_train_frames[1]]

        if len(self.all_id_list) > 2900:  # neural point-based graphics' configuration
            raise NotImplementedError
            self.test_id_list = self.all_id_list[::100]
            self.train_id_list = [
                self.all_id_list[i]
                for i in range(len(self.all_id_list))
                if (
                    ((i % 100) > 19)
                    and (
                        (i % 100) < 81 or (i // 100 + 1) * 100 >= len(self.all_id_list)
                    )
                )
            ]
        else:  # nsvf configuration
            # if self.opt.demo_mode:
            #     test_num = 50
            #     step = 2
            # else:
            #     test_num = -1
            # test_max = 50
            # self.train_id_list = self.all_id_list[::step]
            # self.test_id_list = self.all_id_list[step//2::step*4]
            self.test_id_list = self.all_id_list[::5]
            self.train_id_list = [
                self.all_id_list[i]
                for i in range(len(self.all_id_list))
                if (i % 5) != 0
            ]
            #self.test_id_list = self.test_id_list[:test_num]

        if self.opt.debug_load_n_dataset != -1:
            step = int(len(self.test_id_list) / self.opt.debug_load_n_dataset)
            self.test_id_list = self.test_id_list[::step]
            step = int(len(self.train_id_list) / self.opt.debug_load_n_dataset)
            self.train_id_list = self.train_id_list[::step]


        # print("all_id_list", len(self.all_id_list))
        # print("test_id_list", len(self.test_id_list), self.test_id_list)
        # print("train_id_list", len(self.train_id_list))
        self.train_id_list = self.remove_blurry(self.train_id_list)
        self.id_list = (
            self.train_id_list if self.split == "train" else self.test_id_list
        )
        blur_id = self.detect_blurry(self.id_list)
        if self.split == "train":
            self.id_list = [i for i in self.id_list if i not in blur_id]
        self.view_id_list = []  # ?

    def get_campos_ray(self):
        centerpixel = np.asarray(self.img_wh).astype(np.float32)[None, :] // 2
        camposes = []
        centerdirs = []
        for id in self.id_list:
            c2w = np.loadtxt(
                os.path.join(
                    self.data_dir, self.scan, "pose", "{}.txt".format(id)
                )
            ).astype(
                np.float32
            )  # @ self.blender2opencv
            campos = c2w[:3, 3]
            camrot = c2w[:3, :3]
            raydir = get_dtu_raydir(centerpixel, self.intrinsic, camrot, True)
            camposes.append(campos)
            centerdirs.append(raydir)
        camposes = np.stack(camposes, axis=0)  # 2091, 3
        centerdirs = np.concatenate(centerdirs, axis=0)  # 2091, 3
        # print("camposes", camposes.shape, centerdirs.shape)
        return torch.as_tensor(
            camposes, device="cuda", dtype=torch.float32
        ), torch.as_tensor(centerdirs, device="cuda", dtype=torch.float32)

    def filter_valid_id(self, id_list):
        empty_lst = []
        for id in id_list:
            c2w = np.loadtxt(
                os.path.join(
                    self.root_dir, self.scan, "pose", "{}.txt".format(id)
                )
            ).astype(np.float32)
            if np.max(np.abs(c2w)) < 30:
                empty_lst.append(id)
        return empty_lst

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def detect_blurry(self, list):
        blur_score = []
        for id in list:
            image_path = os.path.join(
                self.root_dir, self.scan, "color/{}.jpg".format(id)
            )
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = self.variance_of_laplacian(gray)
            blur_score.append(fm)
        blur_score = np.asarray(blur_score)
        # less the score is, more blurry the image is
        # TODO: set the value of thresholds
        ids = blur_score.argsort()[:50]
        allind = np.asarray(list)
        print("most blurry images", allind[ids])
        return allind[ids]

    def remove_blurry(self, list):
        blur_path = os.path.join(self.root_dir, self.scan, "blur_list.txt")
        if os.path.exists(blur_path):
            blur_lst = []
            with open(blur_path) as f:
                lines = f.readlines()
                # print("blur files", len(lines))
                for line in lines:
                    info = line.strip()
                    blur_lst.append(int(info))
            return [i for i in list if i not in blur_lst]
        else:
            # print("no blur list detected, use all training frames!")
            return list

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
        self.near_far = [0., 10.]

    def read_points(self):
        if self.split != "train":
            return
        # load point cloud
        if self.opt.demo_mode:
            points_path = "demo/pcd_demo.ply"
        else:
            points_path = "pcd.ply"
        if self.opt.use_comp_points:
            points_path = "pcd_comp.ply"

        points_path = os.path.join(self.root_dir, self.scan, points_path)
        if not os.path.exists(points_path):
            self.parse_mesh()

        points_xyz = read_points_from_path(points_path)

        self.points_xyz_all = points_xyz
        self.init_bbox(points_xyz)
        print(f"points_xyz.shape : {points_xyz.shape}")

    def parse_mesh(self):
        # if pcd.ply does not exist, generate point cloud file from mesh
        if self.opt.demo_mode:
            points_path = os.path.join(self.root_dir, self.scan, "demo/pcd_demo.ply")
            mesh_path = os.path.join(self.root_dir, self.scan, "demo/" + self.scan + "_demo.ply")
        else:
            points_path = os.path.join(self.root_dir, self.scan, "pcd.ply")
            mesh_path = os.path.join(self.root_dir, self.scan, self.scan + "_vh_clean_2.ply")
        plydata = PlyData.read(mesh_path)
        print("plydata 0", plydata.elements[0], plydata.elements[0].data["blue"].dtype)

        vertices = np.empty(
            len(plydata.elements[0].data["blue"]),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        vertices["x"] = plydata.elements[0].data["x"].astype("f4")
        vertices["y"] = plydata.elements[0].data["y"].astype("f4")
        vertices["z"] = plydata.elements[0].data["z"].astype("f4")
        vertices["red"] = plydata.elements[0].data["red"].astype("u1")
        vertices["green"] = plydata.elements[0].data["green"].astype("u1")
        vertices["blue"] = plydata.elements[0].data["blue"].astype("u1")

        # save as ply
        ply = PlyData([PlyElement.describe(vertices, "vertex")], text=False)
        ply.write(points_path)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    # @profile
    def split_by_index(self, index_list):
        index_wh = torch.arange(0, self.img_hw[0] * self.img_hw[1])
        rays_index = (index_list[:, None] + 1) * index_wh[None, :]
        rays_index = rays_index.reshape(-1)
        return self.all_rays[rays_index], self.all_rgbs[rays_index]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.render_train:
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

        elif self.split == "train":  # use data in the buffers
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


if __name__ == "__main__":
    # dataset = ScannetDataset(datadir='/media/data/yxy/ed-nerf/data/scannet/scans/', scan='scene0241_01')
    from opt import config_parser

    args = config_parser()
    dataset = ScannetDataset(
        args.datadir,
        split="train",
        downsample=args.downsample_train,
        is_stack=False,
        margin=args.scannet_margin,
        points_generation_mode=args.points_generation_mode,
        args=args,
    )
    item = dataset.__getitem__(0)
    for key, value in item.items():
        if type(value) == torch.Tensor:
            print(f"key:{key} tensor.shape:{value.shape}")
        else:
            print(f"key:{key} value:{value.shape}")

    print(f"rays.shape {dataset.all_rays.shape}")  # [640000, 6]

    print(f"rgbs.shape : {dataset.all_rgbs.shape}")  # [640000, 3]
