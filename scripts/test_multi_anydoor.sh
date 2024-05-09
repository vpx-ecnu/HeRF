SCENE_NAME=$1
python train.py --config configs/param_exp/scannet/paper_01/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_80k_$SCENE_NAME/step_80k_$SCENE_NAME.th \
--render_only 1 --render_test 1 \
--datadir /media/data/scannet_3dv/scans/$SCENE_NAME \
--expname step_80k_$SCENE_NAME
