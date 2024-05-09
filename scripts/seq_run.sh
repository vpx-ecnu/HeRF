#!/bin/bash

# 定义文件夹路径和配置文件路径
SCANS_DIR="/media/data/scannet_3dv/scans"
CONFIG_FILE="/home/xyyang/code/TensoRF/configs/param_exp/scannet/paper_01/step_80k.txt"

# 遍历文件名在指定范围内的文件
for scene_num in $(seq -f "%04g" 474 592); do
    FILE="$SCANS_DIR/scene${scene_num}_00"
    if [[ -f "$FILE" ]]; then
        # 修改 CONFIG_FILE 中的 datadir 和 expname
        sed -i "s|datadir = .*|datadir = $FILE|g" $CONFIG_FILE
        sed -i "s|expname = .*|expname = step_80k_scene${scene_num}_00|g" $CONFIG_FILE

        # 执行 python 脚本
        python /home/xyyang/code/TensoRF/train.py --config $CONFIG_FILE
    fi
done
