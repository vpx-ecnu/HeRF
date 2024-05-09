import os

import numpy as np
import torch
from tqdm.auto import tqdm

from dataLoader.pose_utils import *
from opt import config_parser


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import pynvml

from dataLoader import dataset_dict
from extra.sparse_scene import SScene
import sys

# from memory_profiler import profile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

env_vars = os.environ
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES"))
gpu_id = 0 if gpu_id is None else gpu_id
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005
    )


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir,
        split="test",
        downsample=args.downsample_test,
        is_stack=True,
        margin=args.scannet_margin,
        points_generation_mode=args.points_generation_mode,
        args=args,
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!!")
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.opt = args
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir,
            split="train",
            downsample=args.downsample_train,
            is_stack=True,
            margin=args.scannet_margin,
            points_generation_mode=args.points_generation_mode,
            args=args,
        )
        PSNRs_test = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/{args.expname}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f"{logfolder}/{args.expname}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/{args.expname}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


@torch.no_grad()
def demo_mode_render(args):
    assert len(args.demo_scene_ckpts) > 0
    renderer = OctreeRender_trilinear_fast_demo

    # load traj points
    # door_pos = np.loadtxt(os.path.join(args.datadir, "demo/door_pos.txt"))
    # door_center = door_pos.mean(axis=0)
    #
    # traj_pnt_1 = np.loadtxt(os.path.join(args.datadir, "demo/traj.txt"))
    # traj_pnt_1 = np.concatenate((door_center[None, :], traj_pnt_1))
    # traj_pnt_2 = np.loadtxt(os.path.join(args.demo_scene_datadir[0], "demo/traj.txt"))
    # traj_pnt_2 = np.concatenate((door_center[None, :], traj_pnt_2))
    # traj_pnt_all = np.concatenate((traj_pnt_1, traj_pnt_2))
    # traj_pnt_all[:, 2] = door_center[2] * 2 * 0.7
    #
    # c2w_list = []
    # for i in range(len(traj_pnt_all) - 1):
    #     c2w = generate_c2w_from_pnts(traj_pnt_all[i], traj_pnt_all[i+1])
    #     c2w_list.append(c2w)
    # c2w_list.append(generate_c2w_from_pnts(traj_pnt_all[-1], traj_pnt_all[0]))
    #
    # c2w_list = interp_poses(c2w_list, 15)
    # c2ws = np.stack(c2w_list)
    # del c2w_list

    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir,
        split="test",
        downsample=args.downsample_test,
        is_stack=True,
        margin=args.scannet_margin,
        points_generation_mode=args.points_generation_mode,
        args=args,
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    # FIXME: traj points hardcoded here
    # read poses from the dataset of first scene
    ids = range(0, 2)
    c2ws_l = []
    for i in ids:
        pose = np.loadtxt(
            os.path.join(
                test_dataset.root_dir, test_dataset.scan, "pose", "{}.txt".format(i)
            )
        ).astype(np.float32)
        print("original pose: ", pose)
        pose = test_dataset.demo_T @ pose if test_dataset.opt.demo_mode else pose
        print("trans pose:", pose)
        c2ws_l.append(pose)
    c2ws_l_rev = c2ws_l[::-1]
    c2ws = np.stack(c2ws_l_rev + c2ws_l)


    tensorf_list = []
    for ckpt_path in args.demo_scene_ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        # ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        # kwargs.update({"device": torch.device("cpu")})
        kwargs["args"] = args
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
        tensorf_list.append(tensorf)

    logfolder = os.path.dirname(args.demo_scene_ckpts[0])

    os.makedirs(f"{logfolder}/imgs_demo_all", exist_ok=True)
    print(logfolder)
    render_demo(
        test_dataset,
        tensorf_list,
        c2ws,
        renderer,
        f"{logfolder}/imgs_demo_all/",
        N_vis=-1,
        N_samples=-1,
        white_bg=white_bg,
        ndc_ray=ndc_ray,
        device=device,
    )


# @profile
def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    # whole dataset
    train_dataset = dataset(
        args.datadir,
        split="train",
        downsample=args.downsample_train,
        is_stack=False,
        margin=args.scannet_margin,
        points_generation_mode=args.points_generation_mode,
        args=args,
    )
    if args.render_path or args.render_test:
        test_dataset = dataset(
            args.datadir,
            split="test",
            downsample=args.downsample_test,
            is_stack=True,
            margin=args.scannet_margin,
            points_generation_mode=args.points_generation_mode,
            args=args,
        )

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # copy the config file into the log folder
    os.system(f"cp {args.config} {logfolder}")

    sscene = SScene(train_dataset)
    sscene.init_scene_bbox()
    sscene.build_blocks(args.block_num_on_ray, args.n_block_per_axis)
    # sscene.build_tensors(args, n_lamb_sigma, n_lamb_sh)
    sscene.split_dataset(train_dataset, args.camera_group_num)

    n_train_split = len(sscene.train_split_dataset)
    print(f"n_train_split: {n_train_split}")

    upsamp_full_list = torch.tensor([0] + upsamp_list + [args.n_iters])
    chunk_interval = upsamp_full_list[1:] - upsamp_full_list[:-1]
    chunk_iter = torch.arange(1, n_train_split)
    chunk_list = chunk_iter[None, :] * chunk_interval[:, None]
    chunk_list = (chunk_list / n_train_split).to(torch.int)
    chunk_list = chunk_list + upsamp_full_list[:-1][:, None]
    chunk_list = chunk_list.flatten()
    print(f"chunk_list: {chunk_list}")

    block_all = sscene.block_list[0].deepcopy()
    block_all = block_all.cat(sscene.block_list[1:])

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = args.n_block_per_axis * args.voxel_num
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # delete origin train_dataset
    # import sys, gc
    # gc.collect()
    # print("!!!!!!!!!!! train_dataset getrefcount:", sys.getrefcount(train_dataset))
    # print(gc.get_referrers(train_dataset))
    # for attr in dir(train_dataset):
    #     if not attr.startswith('__') and not attr.startswith('_'):
    #         delattr(train_dataset, attr)
    del train_dataset.all_light_idx
    del train_dataset.all_masks
    del train_dataset.all_rays
    del train_dataset.all_rgbs
    del train_dataset.directions
    del train_dataset.points_xyz_all
    del train_dataset

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb,
            args.voxel_num,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
            block=block_all,
            args=args,
        )

    # linear in logrithmic space
    # N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]
    # N_voxel_list = torch.linspace(args.voxel_num[0], args.voxel_num_final[0], len(upsamp_list) + 1)
    N_voxel_list = torch.exp(
        torch.linspace(
            np.log(args.voxel_num[0]),
            np.log(args.voxel_num_final[0]),
            len(upsamp_list) + 1,
        )
    )
    N_voxel_list = (N_voxel_list / sscene.block_list[0].N_batch_num).round()
    N_voxel_list = N_voxel_list.round().long().tolist()
    n_voxels = N_voxel_list.pop(0)

    i = 0
    # for i in range(n_train_split):
    allrays, allrgbs, blocks, tensors, block_id_list = sscene.train_split_dataset[i]
    tensorf.assign_block(blocks, tensors, device)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

        # allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(
        f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}"
    )

    pbar = tqdm(
        range(
            0,
            args.n_iters,
        ),
        miniters=args.progress_refresh_rate,
        file=sys.stdout,
    )
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train,
            tensorf,
            chunk=args.batch_size,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            is_train=True,
        )

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar(
                "train/reg", loss_reg.detach().item(), global_step=iteration
            )
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar(
                "train/reg_l1", loss_reg_L1.detach().item(), global_step=iteration
            )

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_density",
                loss_tv.detach().item(),
                global_step=iteration,
            )
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
            )

        optimizer.zero_grad()
        total_loss.backward()
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        optimizer.step()

        loss = loss.detach().item()

        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_used_pynvml = info.used

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar("train/mse", loss, global_step=iteration)
        summary_writer.add_scalar(
            "train/GPU_mem", current_memory / 1024 / 1024, global_step=iteration
        )
        summary_writer.add_scalar(
            "train/GPU_mem_max", max_memory / 1024 / 1024, global_step=iteration
        )
        summary_writer.add_scalar(
            "train/GPU_mem_pynvml",
            gpu_used_pynvml / 1024 / 1024,
            global_step=iteration,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Iteration {iteration:05d}:"
                + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                + f" mse = {loss:.6f}"
            )
            PSNRs = []

        if iteration == int(args.n_iters * 0.8):
            torch.cuda.empty_cache()

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            # sscene.assign_block(tensorf)
            test_blocks = sscene.block_list
            test_tensors = None
            tensorf.assign_block(test_blocks, test_tensors, device, n_voxels)
            PSNRs_test = evaluation(
                test_dataset,
                tensorf,
                args,
                renderer,
                f"{logfolder}/imgs_vis/",
                N_vis=args.N_vis,
                prtx=f"{iteration:06d}_",
                N_samples=nSamples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                compute_extra_metrics=False,
            )
            summary_writer.add_scalar(
                "test/psnr", np.mean(PSNRs_test), global_step=iteration
            )

        if iteration in update_AlphaMask_list:
            if (
                reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
            ):  # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in chunk_list: # and iteration not in upsamp_list:
            # print("training split", i, "of", n_train_split, "splits")
            del allrays, allrgbs
            i = (i + 1) % n_train_split
            allrays, allrgbs, blocks, _, block_id_list = sscene.train_split_dataset[i]
            tensorf.assign_block(blocks, None, device)
            tensorf.upsample_volume_grid(n_voxels)
            if not args.ndc_ray:
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
            trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            # reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            # nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(n_voxels)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(
                args.lr_init * lr_scale, args.lr_basis * lr_scale
            )
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        del rays_train, rgb_train

    tensorf.save(f"{logfolder}/{args.expname}.th")

    tensorf.assign_block(sscene.block_list, None, device)

    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir,
            split="train",
            downsample=args.downsample_train,
            is_stack=True,
            margin=args.scannet_margin,
            points_generation_mode=args.points_generation_mode,
            args=args,
        )
        PSNRs_test = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        tensorf.upsample_volume_grid(n_voxels)
        PSNRs_test = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        summary_writer.add_scalar(
            "test/psnr_all", np.mean(PSNRs_test), global_step=iteration
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        tensorf.upsample_volume_grid(n_voxels)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    elif args.demo_mode_render:
        demo_mode_render(args)
    else:
        reconstruction(args)
    pynvml.nvmlShutdown()
