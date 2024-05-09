python train.py \
--config configs/param_exp/scannet/paper_01/step_1k.txt \
--datadir /media/data/scannet_3dv/scans/scene0198_00 \
--expname step_1k_scene0198_00
# 22.46 21.0
python train.py \
--render_only 1 \
--render_test 1 \
--config configs/param_exp/scannet/paper_01/step_1k.txt \
--datadir /media/data/scannet_3dv/scans/scene0198_00 \
--expname step_1k_scene0198_00 \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_1k_scene0198_00/step_1k_scene0198_00.th


# 133,