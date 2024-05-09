import os, math
import numpy as np
import scipy.signal
from typing import List, Optional
from PIL import Image
import os
import torch
import configargparse
from tqdm import tqdm, trange

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="/media/data/yxy/ed-nerf/data/scannet/scans/scene0241_01/exported",
                        help="gt dir")
    parser.add_argument("--exp_dir", type=str, default="/home/ccg/NeRFusion/results/scannet/bo233_scene0241_01",
                        help="exp dir")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    exp_dir = args.exp_dir
    dataname = os.path.basename(gt_dir)
    expname = os.path.basename(args.exp_dir)
    # img_dir = os.path.join(exp_dir, 'imgs_test_all')
    img_dir = exp_dir

    img_names = list(filter(lambda x: x.endswith('.png'), os.listdir(img_dir)))
    n = len(img_names)
    # fileNum = 80
    test_id = np.loadtxt(f'{gt_dir}/test.txt', dtype=np.int32)

    outFile = f'/home/dblu/metrics233.txt'

    with open(outFile, 'w') as f:
        all_psnr = []
        all_ssim = []
        all_alex = []
        all_vgg = []

        gtstr = gt_dir + "/color/%d.jpg"
        print("expname: ", expname)
        resultstr = exp_dir + "/%03d.png"
        exist_metric=False

        test = [i for i in range(2000) if (i % 5) != 0]

        unpad = 24
        H, W = 968 - 2 * unpad, 1296 - 2 * unpad

        left = unpad
        top = unpad
        right = 1296 - unpad
        bottom = 968 - unpad

        if not exist_metric:
            psnrs = []
            ssims = []
            l_alex = []
            l_vgg = []
            for i in trange(1169):
                gt = Image.open(gtstr%(test[i]))
                gt = gt.crop((left, top, right, bottom))
                width, height = gt.size
                # gt = gt.resize((width // 2, height // 2),
                #                  Image.LANCZOS,
                #                  )
                gt = np.asarray(gt,dtype=np.float32) / 255.0
                # gt = gt[unpad:-unpad, unpad:-unpad,:]
                # gtmask = gt[...,[3]]
                gt = gt[...,:3]


                # gt = gt*gtmask + (1-gtmask)
                img = np.asarray(Image.open(resultstr%i),dtype=np.float32)[...,:3]  / 255.0
                # print(gt[0,0],img[0,0],gt.shape, img.shape, gt.max(), img.max())

                img = torch.from_numpy(img)
                img = torch.nn.functional.interpolate(
                    img.permute(2, 0, 1)[None, ...],
                    size=(gt.shape[0], gt.shape[1]),
                    mode="bilinear",
                    align_corners=True
                    # mode="nearest",
                )[0].permute(1,2,0).numpy()
                psnr = -10. * np.log10(np.mean(np.square(img - gt)))
                ssim = rgb_ssim(img, gt, 1)
                lpips_alex = rgb_lpips(gt, img, 'alex','cuda')
                # lpips_vgg = rgb_lpips(gt, img, 'vgg','cuda')
                lpips_vgg = 0

                print(i, psnr, ssim, lpips_alex, lpips_vgg)
                f.write(f'{i} : psnr {psnr} ssim {ssim}  l_a {lpips_alex} l_v {lpips_vgg}\n')
                psnrs.append(psnr)
                ssims.append(ssim)
                l_alex.append(lpips_alex)
                l_vgg.append(lpips_vgg)
                psnr = np.mean(np.array(psnrs))
                ssim = np.mean(np.array(ssims))
                l_a  = np.mean(np.array(l_alex))
                l_v  = np.mean(np.array(l_vgg))

        rS=f'{dataname} : psnr {psnr} ssim {ssim}  l_a {l_a} l_v {l_v}'
        print(rS)
        f.write(rS+"\n")

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_alex.append(l_a)
        all_vgg.append(l_v)

        psnr = np.mean(np.array(all_psnr))
        ssim = np.mean(np.array(all_ssim))
        l_a  = np.mean(np.array(all_alex))
        l_v  = np.mean(np.array(all_vgg))

        rS=f'mean : psnr {psnr} ssim {ssim}  l_a {l_a} l_v {l_v}'
        print(rS)
        f.write(rS+"\n")
