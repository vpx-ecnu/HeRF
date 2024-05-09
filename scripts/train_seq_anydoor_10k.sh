#
#scenes=( "scene0178_00" "scene0333_00" "scene0418_00" "scene0155_00" "scene0133_00")
## scenes=("scene0417_00" "scene0298_00" "scene0178_00" "scene0333_00" "scene0418_00" "scene0155_00" "scene0133_00" "scene0156_00" )
#steps=("10k" "1k" "30k")
#
#for scene in "${scenes[@]}"
#do
#    for step in "${steps[@]}"
#    do
#        echo "==========================================="
#        echo "train $scene $step"
#        bash scripts/train_multi_anydoor_10k.sh $scene $step
#        echo "==========================================="
#        echo "test $scene $step"
#        bash scripts/test_multi_anydoor_10k.sh $scene $step
#        echo "==========================================="
#    done
#done

bash scripts/train_multi_anydoor_10k.sh scene0417_00 80k