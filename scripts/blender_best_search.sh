python train.py \
--config configs/param_exp/scannet/paper_01/step_1k.txt \
--datadir /media/data/scannet_3dv/scans/scene0198_00 \
--expname step_1k_scene0198_00
# voxel num, batch voxel, step_ratio, upsample time, lr, L1_weight_inital, density detach
python train.py \
--render_only 1 \
--render_test 1 \
--config configs/param_exp/scannet/paper_01/step_1k.txt \
--datadir /media/data/scannet_3dv/scans/scene0178_00 \
--expname step_80k_scene0178_00 \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_80k_scene0178_00/step_80k_scene0178_00.th


# 133, 23.7