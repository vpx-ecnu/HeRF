#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

#run="taskset -c 32,33,34,35,36,37,38,39"
run="4taskset -c 40,41,42,43,44,45,46,47"
#run="taskset -c 48,49,50,51,52,53,54,55"
#run="taskset -c 56,57,58,59,60,61,62,63"





${run} \
python train.py \
--config \
/media/data/yxy/ed-nerf/logs/tensoir/log_demo/0816_demo_scene0336_00_no_pad-20230816-120104/train_init_demo.txt \
--demo_mode \
--demo_mode_render \
--demo_scene_ckpts \
/media/data/yxy/ed-nerf/logs/tensoir/log_demo/0816_demo_scene0336_00_no_pad-20230816-120104/0816_demo_scene0336_00_no_pad.th \
/media/data/yxy/ed-nerf/logs/tensoir/log_demo/0816_demo_scene0369_00_part-20230816-120311/0816_demo_scene0369_00_part.th \
--demo_scene_datadir \
/media/data/dblu/datasets/scannet/scans/demo/scene0101_04_win8_rev \
;


#${run} \
#python train.py \
#--config \
#/media/data/dblu/logs/TensoRF/demo/0814_win_120k_2-20230814-102151/scene0241_01_win2_120k.txt \
#--demo_mode \
#--demo_mode_render \
#--demo_scene_ckpts \
#/media/data/dblu/logs/TensoRF/demo/0814_win_120k_2-20230814-102151/0814_win_120k_2.th \
#/media/data/dblu/logs/TensoRF/demo/0814_scene0149_00_win0_10k-20230814-213218/0814_scene0149_00_win0_10k.th \
#--demo_scene_datadir \
#/media/data/dblu/datasets/scannet/scans/demo/scene0101_04_win8_rev \
#;