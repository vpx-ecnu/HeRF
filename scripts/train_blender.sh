
#scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" 'ship')
scenes=("materials")
steps=("30k")

for scene in "${scenes[@]}"
do
    for step in "${steps[@]}"
    do
        python train.py --config configs/param_exp/blender/paper_01/step_$step.txt \
        --datadir /media/data/yxy/ed-nerf/data/nerf/nerf_synthetic/$scene \
        --expname step_${step}_${scene}
    done
done

