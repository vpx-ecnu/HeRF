# HeRF: A Hierarchical Framework for Efficient and Extendable New View Synthesis  

Xiaoyan Yang, Dingbo Lu, Wenjie Liu, Ling You, Yang Li, Changbo Wang

IJCNN 2024 paper(coming soon),  [Project Page](https://vpx-ecnu.github.io/HeRF_website/), Video(coming soon)

## Installation



We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Run the following commands:
```shell
# Clone the repo
git clone https://github.com/Minisal/HeRF.git
cd HeRF
# Create a conda environment
conda create --name herf python=3.9.12
conda activate herf
# Prepare pip
conda install pip
pip install --upgrade pip
# install cudatoolkit, otherwise you can config it to your cuda path
conda install cudatoolkit=11.3
# Install PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Install other
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```


## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
* [Scannet](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

[//]: # (* [Synthetic-NSVF]&#40;https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip&#41;)

[//]: # (* [Tanks&Temples]&#40;https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip&#41;)

[//]: # (* [Forward-facing]&#40;https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1&#41;)



## Quick Start
The training script is in `train.py`, to train:

```
python train.py --config configs/param_exp/scannet/paper_01/step_1k.txt
```



## Rendering

```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder. 

## Extracting mesh
You can also export the mesh by passing `--export_mesh 1`:
```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --export_mesh 1
```
Note: Please re-train the model and don't use the pretrained checkpoints provided by us for mesh extraction, 
because some render parameters has changed.


## Citation 
If you find our code or paper helps, please consider citing:
```
@article{yang2024ijcnn,
    title={{HeRF}: A Hierarchical Framework for Efficient and Extendable New View Synthesis},
    author={Xiaoyan Yang, Dingbo Lu, Wenjie Liu, Ling You, Yang Li, Changbo Wang},
    journal={IJCNN},
    year={2024}
}
```
