SCENE_NAME=$1
STEP=$2
python train.py --config configs/param_exp/scannet/paper_01/step_${STEP}.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_${STEP}_${SCENE_NAME}/step_${STEP}_${SCENE_NAME}.th \
--render_only 1 --render_test 1 \
--datadir /media/data/scannet_3dv/scans/$SCENE_NAME \
--expname step_${STEP}_${SCENE_NAME}
