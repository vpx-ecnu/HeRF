import os
import torch
import numpy as np
import open3d as o3d
import pickle
from tqdm import tqdm
from utils import timer
from models.pointnerf.mvs_points_volumetric_model import MvsPointsVolumetricModel
from models.pointnerf.mvs import mvs_utils, filter_utils
from models.pointnerf.visualizer import Visualizer

# from utils import format as fmt
# from run.evaluate import report_metrics


torch.manual_seed(0)
np.random.seed(0)

DEBUG = True
DEBUG_SAVE = True


def get_coordinate_plane_and_line(
    xyz_sampled, matMode, vecMode, kdtree=None, aabb=None, detach=True
):
    index_mode = "kdtree" if kdtree is not None else "tensor"
    assert index_mode in ["tensor", "kdtree"]

    if DEBUG:
        xyz_sampled = xyz_sampled[:100]

    if index_mode == "tensor":
        raise ValueError("index mode should not be tensor")
        index = xyz_sampled

    if index_mode == "kdtree":
        N = xyz_sampled.shape[0]
        xyz_sampled = xyz_sampled
        vox_index = xyz_sampled.view(1, N, 1, 1, 3)
        index = (
            torch.nn.functional.grid_sample(
                kdtree.cache[None, ...], vox_index, mode="nearest"
            )
            * 2.0
            - 1.0
        )
        index = index.view(3, N).T

        if DEBUG_SAVE:
            invaabbSize = 2.0 / (aabb[1] - aabb[0])
            xyz_sampled = (xyz_sampled + 1) / invaabbSize + aabb[0]
            save_dict = {
                "xyz_sampled": xyz_sampled,
                "index": index,
            }
            save(
                save_dict,
                "/home/dblu/data/logs/TensoRF/debug/get_coordinate_plane_and_line.pkl",
            )
            raise ValueError("debug save")

    coordinate_plane = torch.stack(
        (index[..., matMode[0]], index[..., matMode[1]], index[..., matMode[2]])
    ).view(3, -1, 1, 2)
    coordinate_line = torch.stack(
        (index[..., vecMode[0]], index[..., vecMode[1]], index[..., vecMode[2]])
    )
    coordinate_line = torch.stack(
        (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
    ).view(3, -1, 1, 2)

    if detach:
        coordinate_plane = coordinate_plane.detach()
        coordinate_line = coordinate_line.detach()
    return coordinate_plane, coordinate_line


@timer
def save(file, filepath):
    print("saving to", filepath)
    save_path = os.path.split(filepath)[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(filepath, "wb") as f:
        pickle.dump(file, f)


@timer
def load(filepath):
    print("loading from", filepath)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [
        item[:, mask, ...] if item is not None else None for item in seconddim_lst
    ]
    return first_lst, second_lst


def triangluation_bpa(pnts, test_pnts=None, full_comb=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(
        pnts[:, :3] / np.linalg.norm(pnts[:, :3], axis=-1, keepdims=True)
    )

    # pcd.colors = o3d.utility.Vector3dVector(pnts[:, 3:6] / 255)
    # pcd.normals = o3d.utility.Vector3dVector(pnts[:, 6:9])
    # o3d.visualization.draw_geometries([pcd])

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radius = 3 * avg_dist
    dec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    # dec_mesh = dec_mesh.simplify_quadric_decimation(100000)
    # dec_mesh.remove_degenerate_triangles()
    # dec_mesh.remove_duplicated_triangles()
    # dec_mesh.remove_duplicated_vertices()
    # dec_mesh.remove_non_manifold_edges()

    # vis_lst = [dec_mesh, pcd]
    # vis_lst = [dec_mesh, pcd]
    # o3d.visualization.draw_geometries(vis_lst)
    # if test_pnts is not None :
    #     tpcd = o3d.geometry.PointCloud()
    #     print("test_pnts",test_pnts.shape)
    #     tpcd.points = o3d.utility.Vector3dVector(test_pnts[:, :3])
    #     tpcd.normals = o3d.utility.Vector3dVector(test_pnts[:, :3] / np.linalg.norm(test_pnts[:, :3], axis=-1, keepdims=True))
    #     o3d.visualization.draw_geometries([dec_mesh, tpcd] )
    triangles = np.asarray(dec_mesh.triangles, dtype=np.int32)
    if full_comb:
        q, w, e = triangles[..., 0], triangles[..., 1], triangles[..., 2]
        triangles2 = np.stack([w, q, e], axis=-1)
        triangles3 = np.stack([e, q, w], axis=-1)
        triangles = np.concatenate([triangles, triangles2, triangles3], axis=0)
    return triangles


def gen_points_filter_embeddings(dataset, opt):
    print(
        "-----------------------------------Generate Points-----------------------------------"
    )

    # opt.is_train = False
    # opt.mode = 1
    visualizer = Visualizer(opt)

    # MvsPointsVolumetricModel.modify_commandline_options(opt, is_train=False)
    model = MvsPointsVolumetricModel()
    model.is_train = False
    model.initialize(opt)
    model.setup(opt)
    model.eval()

    cam_xyz_all = []
    intrinsics_all = []
    extrinsics_all = []
    confidence_all = []
    points_mask_all = []
    intrinsics_full_lst = []
    confidence_filtered_all = []
    near_fars_all = []
    gpu_filter = True
    cpu2gpu = len(dataset.view_id_list) > 300

    imgs_lst, HDWD_lst, c2ws_lst, w2cs_lst, intrinsics_lst = [], [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.view_id_list))):
            data = dataset.get_init_item(i)
            model.set_input(data)
            # intrinsics    1, 3, 3, 3
            (
                points_xyz_lst,
                photometric_confidence_lst,
                point_mask_lst,
                intrinsics_lst,
                extrinsics_lst,
                HDWD,
                c2ws,
                w2cs,
                intrinsics,
                near_fars,
            ) = model.gen_points()
            # visualizer.save_neural_points(i, points_xyz_lst[0].view(-1, 3), None, data, save_ref=opt.load_points == 0)
            B, N, C, H, W, _ = points_xyz_lst[0].shape
            # print("points_xyz_lst",points_xyz_lst[0].shape)
            cam_xyz_all.append(
                (points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0])
                if gpu_filter
                else points_xyz_lst[0].cpu().numpy()
            )
            # intrinsics_lst[0] 1, 3, 3
            intrinsics_all.append(
                intrinsics_lst[0] if gpu_filter else intrinsics_lst[0]
            )
            extrinsics_all.append(
                extrinsics_lst[0] if gpu_filter else extrinsics_lst[0].cpu().numpy()
            )
            if opt.manual_depth_view != 0:
                confidence_all.append(
                    (
                        photometric_confidence_lst[0].cpu()
                        if cpu2gpu
                        else photometric_confidence_lst[0]
                    )
                    if gpu_filter
                    else photometric_confidence_lst[0].cpu().numpy()
                )
            points_mask_all.append(
                (point_mask_lst[0].cpu() if cpu2gpu else point_mask_lst[0])
                if gpu_filter
                else point_mask_lst[0].cpu().numpy()
            )
            imgs_lst.append(data["images"].cpu())
            HDWD_lst.append(HDWD)
            c2ws_lst.append(c2ws)
            w2cs_lst.append(w2cs)
            intrinsics_full_lst.append(intrinsics)
            near_fars_all.append(near_fars[0, 0])
            # visualizer.save_neural_points(i, points_xyz_lst[0].view(-1, 3), None, data, save_ref=opt.load_points == 0)
            # #################### start query embedding ##################
        torch.cuda.empty_cache()
        if opt.manual_depth_view != 0:
            if gpu_filter:
                (
                    _,
                    xyz_world_all,
                    confidence_filtered_all,
                ) = filter_utils.filter_by_masks_gpu(
                    cam_xyz_all,
                    intrinsics_all,
                    extrinsics_all,
                    confidence_all,
                    points_mask_all,
                    opt,
                    vis=True,
                    return_w=True,
                    cpu2gpu=cpu2gpu,
                    near_fars_all=near_fars_all,
                )
            else:
                (
                    _,
                    xyz_world_all,
                    confidence_filtered_all,
                ) = filter_utils.filter_by_masks(
                    cam_xyz_all,
                    [intr.cpu().numpy() for intr in intrinsics_all],
                    extrinsics_all,
                    confidence_all,
                    points_mask_all,
                    opt,
                )
            # print(xyz_ref_lst[0].shape) # 224909, 3
        else:
            cam_xyz_all = [
                cam_xyz_all[i].reshape(-1, 3)[points_mask_all[i].reshape(-1), :]
                for i in range(len(cam_xyz_all))
            ]
            xyz_world_all = [
                np.matmul(
                    np.concatenate(
                        [cam_xyz_all[i], np.ones_like(cam_xyz_all[i][..., 0:1])],
                        axis=-1,
                    ),
                    np.transpose(np.linalg.inv(extrinsics_all[i][0, ...])),
                )[:, :3]
                for i in range(len(cam_xyz_all))
            ]
            (
                xyz_world_all,
                cam_xyz_all,
                confidence_filtered_all,
            ) = filter_utils.range_mask_lst_np(
                xyz_world_all, cam_xyz_all, confidence_filtered_all, opt
            )
            del cam_xyz_all
        # for i in range(len(xyz_world_all)):
        #    visualizer.save_neural_points(i, torch.as_tensor(xyz_world_all[i], device="cuda", dtype=torch.float32), None, data, save_ref=opt.load_points==0)
        # exit()
        # xyz_world_all = xyz_world_all.cuda()
        # confidence_filtered_all = confidence_filtered_all.cuda()
        points_vid = torch.cat(
            [
                torch.ones_like(xyz_world_all[i][..., 0:1]) * i
                for i in range(len(xyz_world_all))
            ],
            dim=0,
        )
        xyz_world_all = (
            torch.cat(xyz_world_all, dim=0)
            if gpu_filter
            else torch.as_tensor(
                np.concatenate(xyz_world_all, axis=0),
                device="cuda",
                dtype=torch.float32,
            )
        )
        confidence_filtered_all = (
            torch.cat(confidence_filtered_all, dim=0)
            if gpu_filter
            else torch.as_tensor(
                np.concatenate(confidence_filtered_all, axis=0),
                device="cuda",
                dtype=torch.float32,
            )
        )
        print(
            "xyz_world_all",
            xyz_world_all.shape,
            points_vid.shape,
            confidence_filtered_all.shape,
        )
        torch.cuda.empty_cache()
        # visualizer.save_neural_points(0, xyz_world_all, None, None, save_ref=False)
        # print("vis 0")

        print(
            "%%%%%%%%%%%%%  getattr(dataset, spacemin, None)",
            getattr(dataset, "spacemin", None),
        )
        if getattr(dataset, "spacemin", None) is not None:
            mask = (
                xyz_world_all - dataset.spacemin[None, ...].to(xyz_world_all.device)
            ) >= 0
            mask *= (
                dataset.spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all
            ) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(
                mask, [xyz_world_all, points_vid, confidence_filtered_all], []
            )
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
        # visualizer.save_neural_points(50, xyz_world_all, None, None, save_ref=False)
        # print("vis 50")
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(
                xyz_world_all,
                dataset.alphas,
                dataset.intrinsics,
                dataset.cam2worlds,
                dataset.world2cams,
                dataset.near_far
                if opt.ranges[0] < -90.0 and getattr(dataset, "spacemin", None) is None
                else None,
                opt=opt,
            )
            first_lst, second_lst = masking(
                vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], []
            )
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)
        # visualizer.save_neural_points(100, xyz_world_all, None, data, save_ref=opt.load_points == 0)
        # print("vis 100")

        if opt.vox_res > 0:
            (
                xyz_world_all,
                sparse_grid_idx,
                sampled_pnt_idx,
            ) = mvs_utils.construct_vox_points_closest(
                xyz_world_all.cuda()
                if len(xyz_world_all) < 99999999
                else xyz_world_all[:: (len(xyz_world_all) // 99999999 + 1), ...].cuda(),
                opt.vox_res,
            )
            points_vid = points_vid[sampled_pnt_idx, :]
            confidence_filtered_all = confidence_filtered_all[sampled_pnt_idx]
            print("after voxelize:", xyz_world_all.shape, points_vid.shape)
            xyz_world_all = xyz_world_all.cuda()

        xyz_world_all = [
            xyz_world_all[points_vid[:, 0] == i, :] for i in range(len(HDWD_lst))
        ]
        confidence_filtered_all = [
            confidence_filtered_all[points_vid[:, 0] == i] for i in range(len(HDWD_lst))
        ]
        cam_xyz_all = [
            (
                torch.cat(
                    [xyz_world_all[i], torch.ones_like(xyz_world_all[i][..., 0:1])],
                    dim=-1,
                )
                @ extrinsics_all[i][0].t()
            )[..., :3]
            for i in range(len(HDWD_lst))
        ]
        points_embedding_all, points_color_all, points_dir_all, points_conf_all = (
            [],
            [],
            [],
            [],
        )
        for i in tqdm(range(len(HDWD_lst))):
            if len(xyz_world_all[i]) > 0:
                embedding, color, dir, conf = model.query_embedding(
                    HDWD_lst[i],
                    torch.as_tensor(
                        cam_xyz_all[i][None, ...], device="cuda", dtype=torch.float32
                    ),
                    torch.as_tensor(
                        confidence_filtered_all[i][None, :, None],
                        device="cuda",
                        dtype=torch.float32,
                    )
                    if len(confidence_filtered_all) > 0
                    else None,
                    imgs_lst[i].cuda(),
                    c2ws_lst[i],
                    w2cs_lst[i],
                    intrinsics_full_lst[i],
                    0,
                    pointdir_w=True,
                )
                points_embedding_all.append(embedding)
                points_color_all.append(color)
                points_dir_all.append(dir)
                points_conf_all.append(conf)

        xyz_world_all = torch.cat(xyz_world_all, dim=0)
        points_embedding_all = torch.cat(points_embedding_all, dim=1)
        points_color_all = (
            torch.cat(points_color_all, dim=1)
            if points_color_all[0] is not None
            else None
        )
        points_dir_all = (
            torch.cat(points_dir_all, dim=1) if points_dir_all[0] is not None else None
        )
        points_conf_all = (
            torch.cat(points_conf_all, dim=1)
            if points_conf_all[0] is not None
            else None
        )

        visualizer.save_neural_points(
            200, xyz_world_all, points_color_all, data, save_ref=opt.load_points == 0
        )
        print("vis")
        model.cleanup()
        del model
    return (
        xyz_world_all,
        points_embedding_all,
        points_color_all,
        points_dir_all,
        points_conf_all,
        [img[0].cpu() for img in imgs_lst],
        [c2w for c2w in c2ws_lst],
        [w2c for w2c in w2cs_lst],
        intrinsics_all,
        [list(HDWD) for HDWD in HDWD_lst],
    )
