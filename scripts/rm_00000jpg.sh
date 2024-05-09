#!/bin/bash 
BASE_DIR="/media/data/scannet_3dv/scans"
for dir in ${BASE_DIR}/scene*/; do
	if [[ -d "${dir}color" ]]; then
		 rm ${dir}color/[0-9][0-9][0-9][0-9][0-9].jpg
	fi
done
