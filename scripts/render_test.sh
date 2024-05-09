#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

run="taskset -c 40,41,42,43,44,45,46,47"


python train.py \
--config configs/param_exp/scannet/paper_01/step_80k.txt \
--ckpt /media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_80k_scene0178_00/step_80k_scene0178_00.th \
--render_only 1 --render_test 1