import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True,
                                is_train=False, device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def OctreeRender_trilinear_fast_demo(rays, tensorf_list, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True,
                                     is_train=False, device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        sigma_list, rgb_list, z_vals_list = [], [], []
        for tensorf in tensorf_list:
            sigma, rgb, z_vals, distance_scale = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                            N_samples=N_samples, return_raw=True)
            sigma_list.append(sigma)
            rgb_list.append(rgb)
            z_vals_list.append(z_vals)
        sigma = torch.cat(sigma_list, dim=1)
        rgb = torch.cat(rgb_list, dim=1)
        z_vals = torch.cat(z_vals_list, dim=1)
        _, sorted_idx = torch.sort(z_vals)
        sorted_sigma = torch.gather(sigma, dim=1, index=sorted_idx)
        sorted_rgb = torch.gather(rgb, dim=1, index=sorted_idx.unsqueeze(-1).expand(-1, -1, 3))

        # fixme: fix black hole
        # sorted_sigma = torch.where(sorted_rgb.sum(dim=-1) == 0, 0., sorted_sigma)

        del sigma, sigma_list, rgb, rgb_list

        # real volume render in demo mode
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
            dim=-1,
        )
        alpha, weight, bg_weight = raw2alpha(sorted_sigma, dists * distance_scale)
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * sorted_rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass
    print(f"w, h: {test_dataset.img_wh}")
    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=3072, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        #depth_map_debug = torch.nn.functional.interpolate(depth_map[None, None, ...], (237, 319))[0,0].detach().cpu().numpy()
        #rgb_map_debug = torch.nn.functional.interpolate(rgb_map[None, ...].permute(0, 3, 1, 2), (237, 319))[0].permute(1, 2, 0).detach().cpu().numpy()

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)

            #valid_mask = torch.logical_and(depth_map > near_far[0]+0.01, depth_map < near_far[1]-0.01)
            rgb_map_ = rgb_map.clone()
            #rgb_map_[~valid_mask] = 0.0
            gt_rgb_ = gt_rgb.clone()
            #gt_rgb_[~valid_mask] = 0.0

            loss = torch.mean((rgb_map_ - gt_rgb_) ** 2)
            psnr = -10.0 * np.log(loss.item()) / np.log(10.0)
            PSNRs.append(psnr)
            print("PSNR: ", psnr)
            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map_, gt_rgb_, 1)
                torch.cuda.empty_cache()
                l_a = rgb_lpips(gt_rgb_.numpy(), rgb_map_.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb_.numpy(), rgb_map_.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

        print(f'PSNR: {psnr:.4f}')
        print(f'SSIM: {ssim:.4f}')
        print(f'LPIPS: {l_v:.4f}')

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs


@torch.no_grad()
def render_demo(test_dataset, tensorf_list, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far

    # tensorf_list[0] = tensorf_list[0].cuda()
    img_idx_base = 0
    for scene_i in range(1, len(tensorf_list)):
        render_list = [tensorf_list[0], tensorf_list[scene_i]]
        if scene_i == 1:
            n = c2ws.shape[0]
            render_c2ws = c2ws[n // 2:]
        elif scene_i == len(tensorf_list) - 1:
            n = c2ws.shape[0]
            render_c2ws = c2ws[:n // 2]
        else:
            render_c2ws = c2ws
        for idx, c2w in tqdm(enumerate(render_c2ws)):

            W, H = test_dataset.img_wh

            c2w = torch.FloatTensor(c2w)
            rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
            if ndc_ray:
                rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

            rgb_map, _, depth_map, _, _ = renderer(rays, render_list, chunk=3072, N_samples=N_samples,
                                                   ndc_ray=ndc_ray, white_bg=white_bg, device=device)
            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            if savePath is not None:
                imageio.imwrite(f'{savePath}/{prtx}{(img_idx_base+idx):03d}.png', rgb_map)
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f'{savePath}/rgbd/{prtx}{(img_idx_base+idx):03d}.png', rgb_map)
        img_idx_base += len(render_c2ws)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)
