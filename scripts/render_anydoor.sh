
SCENE_NAME=$1
CONFIG_FILE="configs/param_exp/scannet/paper_01/step_80k.txt"

sed -i "s|datadir = /media/data/scannet_3dv/scans/scene0400_00|datadir = /media/data/scannet_3dv/scans/$SCENE_NAME|g" $CONFIG_FILE
sed -i "s|expname = step_80k_scene0400_00|expname = step_80k_$SCENE_NAME/g" $CONFIG_FILE
sed -i "s|basedir = /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01|basedir = /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/$SCENE_NAME|g" $CONFIG_FILE

sed -i "s|\(datadir = /media/data/scannet_3dv/scans/\)scene0400_00|\1$SCENE_NAME|g" $CONFIG_FILE
sed -i "s|\(expname = step_80k_\)scene0400_00|\1$SCENE_NAME|g" $CONFIG_FILE
sed -i "s|\(basedir = /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/\)scene0400_00|\1$SCENE_NAME|g" $CONFIG_FILE

sed -i 's/data_root = \/data\/scene0*/data_root = \/data\/$SCENE_NAME/g' $CONFIG_FILE

cat $CONFIG_FILE

# 2.
python train.py \
--config configs/param_exp/scannet/paper_01/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_80k_scene0335_01/step_80k_scene0335_01.th \
--render_only 1 --render_test 1
