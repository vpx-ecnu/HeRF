
#python train.py --config configs/param_exp/blender/paper_01/step_80k.txt

python train.py --config configs/param_exp/lidar/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_lidar/0118_house_10k/0118_house_10k.th \
--render_only 1 --render_test 1 \

python train.py --config configs/param_exp/lidar/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_lidar/0118_corner_10k/0118_corner_10k.th \
--render_only 1 --render_test 1 \
--datadir /media/data/dblu/datasets/scannet/scans/corridor_corner/exported \
--expname 0118_corner_10k

python train.py --config configs/param_exp/lidar/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_lidar/0118_stage_10k/0118_stage_10k.th \
--render_only 1 --render_test 1 \
--datadir /media/data/dblu/datasets/scannet/scans/stage/exported \
--expname 0118_stage_10k

python train.py --config configs/param_exp/lidar/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_lidar/0118_huashi_10k/0118_huashi_10k.th \
--render_only 1 --render_test 1 \
--datadir /media/data/dblu/datasets/scannet/scans/huashi/exported \
--expname 0118_huashi_10k
