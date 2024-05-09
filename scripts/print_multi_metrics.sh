scenexxxx_xx=$1
step=$2
echo $scenexxxx_xx
echo $step

directory="/media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/step_${step}_$scenexxxx_xx/step_${step}_$scenexxxx_xx/imgs_test_all/"
files=$(ls "$directory"/*.png | sort -t/ -k2,2n)
max_file=$(echo "$files" | awk -F/ '{print $NF}' | sort -n -t . -k 1,1 -k 2,2 | tail -n 1)
max_index=$(echo "max_file" | grep -oE '[0-9]+')
max_index=$((10#$max_index))
echo $max_index

cat "$directory/mean.txt"
echo "=========================="