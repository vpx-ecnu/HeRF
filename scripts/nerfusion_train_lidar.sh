export PATH="/home/ccg/anaconda3/envs/ngp_pl/bin/:$PATH"
scenexxxx_xx=$1
directory = "/media/data/dblu/datasets/scannet/scans/$scenexxxx_xx/exported/color"
files =$(ls "$directory" / *.jpg | sort -t/ -k2, 2n)
max_file =$(echo "$files" | awk -F/ '{print $NF}' | sort -n -t.-k 1, 1 -k 2, 2 | tail -n 1)
max_index =$(echo "$max_file" | grep -oE '[0-9]+')
max_index =$((10#$max_index-1))
echo $max_index
if["$max_index" -gt 1500]; then
  max_index=1500
fi

python bo233.py $max_index

cp train.txt "/media/data/scannet_3dv/scans/$scenexxxx_xx/"
cp test.txt "/media/data/scannet_3dv/scans/$scenexxxx_xx/"
python scripts/scannet_get_bbox.py "/media/data/scannet_3dv/scans/$scenexxxx_xx/"
python train.py \
  --root_dir "/media/data/scannet_3dv/scans/$scenexxxx_xx" \
  --dataset_name scannet \
  --exp_name "nerfusion_s10_$scenexxxx_xx"
