#!/bin/bash

scene=scene0369_00
python data_preprocess/reader.py \
--filename /media/data/yxy/ed-nerf/data/scannet/demo/scans/${scene}/${scene}.sens \
--output_path /media/data/yxy/ed-nerf/data/scannet/demo/scans/${scene}/exported/ \
--output_path /media/data/yxy/ed-nerf/data/scannet/demo/scans/${scene}/exported/ \
--export_depth_images \
--export_color_images \
--export_poses \
--export_intrinsics \
;