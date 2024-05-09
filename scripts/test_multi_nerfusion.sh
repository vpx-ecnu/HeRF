SCENE_NAME=$1
export PATH="/home/ccg/anaconda3/envs/ngp_pl/bin/:$PATH"

python train.py \
--val_only \
--dataset_name scannet \
--root_dir "/media/data/scannet_3dv/scans/$SCENE_NAME" \
--exp_name "nerfusion_$SCENE_NAME" \
--ckpt_path "ckpts/scannet/nerfusion_$SCENE_NAME/epoch=79.ckpt"