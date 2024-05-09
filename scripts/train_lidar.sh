
#python train.py --config configs/param_exp/blender/paper_01/step_80k.txt

#python train.py --config configs/param_exp/lidar/step_30k.txt \
#--datadir /media/data/dblu/datasets/scannet/scans/house_new/exported \
#--expname lidar_house_30k_step
#
#python train.py --config configs/param_exp/lidar/step_30k.txt \
#--datadir /media/data/dblu/datasets/scannet/scans/corridor_corner/exported \
#--expname lidar_corner_30k_step
##
#python train.py --config configs/param_exp/lidar/step_30k.txt \
#--datadir /media/data/dblu/datasets/scannet/scans/stage/exported \
#--expname lidar_stage_30k_step
##
python train.py --config configs/param_exp/lidar/step_30k.txt \
--datadir /media/data/dblu/datasets/scannet/scans/huashi/exported \
--expname lidar_huashi_30k_step
