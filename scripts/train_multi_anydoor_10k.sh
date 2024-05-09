SCENE_NAME=$1
TRAIN_STEP=$2
python train.py --config configs/param_exp/scannet/paper_01/step_${TRAIN_STEP}.txt \
--datadir /media/data/scannet_3dv/scans/$SCENE_NAME \
--expname step_${TRAIN_STEP}_$SCENE_NAME
