SCENE_NAME=$1
python train.py --config configs/param_exp/scannet/paper_01/step_80k.txt \
--datadir /media/data/scannet_3dv/scans/$SCENE_NAME \
--expname step_80k_$SCENE_NAME
