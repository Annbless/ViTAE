<h1 align="left">ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias <a href="https://arxiv.org/pdf/2106.03348.pdf"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
</a> </h1> 

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#Updates">Updates</a> |
  <a href="#Usage">Usage</a> |
  <a href="#results">Results&Pretrained Models</a> |
  <a href="#statement">Statement</a> |
</p>

## Introduction

<p align="left">This repository contains the code, models, test results for the paper <a href="https://arxiv.org/pdf/2106.03348.pdf">ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias</a>. It contains several reduction cells and normal cells to introduce scale-invariance and locality into vision transformers.

<img src="figs/NetworkStructure.png">

## Updates
***07/12/2021***
The code is released!

***19/10/2021***
The paper is accepted by Neurips'2021! The code will be released soon!
  
***06/08/2021***
The paper is post on arxiv! The code will be made public available once cleaned up.

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/Annbless/ViTAE.git
cd ViTAE
```

- Create a conda virtual environment and activate it:

```bash
conda create -n vitae python=3.7 -y
conda activate vitae
```

```bash
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.2 -c pytorch -c conda-forge
```

- Install `timm==0.3.4`:

```bash
pip install timm==0.3.4
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install pyyaml ipdb
```

### Data Prepare
We use standard ImageNet dataset, you can download it from http://image-net.org/. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

### Evaluation

Take ViTAE_basic_7 as an example, to evaluate the pretrained ViTAE model on ImageNet val, run 

```bash
python validate.py [ImageNetPath] --model ViTAE_basic_7 --eval_checkpoint [Checkpoint Path]
```

### Training

Take ViTAE_basic_7 as an example, to train the ViTAE model on ImageNet with 4 GPU and 512 batch size, run

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py [ImageNetPath] --model ViTAE_basic_7 -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp
```

The trained model file will be saved under the ```output``` folder

## Results

## Main Results on ImageNet-1K with pretrained models
| name | resolution | acc@1 | acc@5 | acc@RealTop-1 | Pretrained |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ViTAE-T | 224x224 | 75.3 | 92.7 | 82.9 | Coming Soon |
| ViTAE-6M | 224x224 | 77.9 | 94.1 | 84.9 | Coming Soon |
| ViTAE-13M | 224x224 | 81.0 | 95.4 | 86.9 | Coming Soon |
| ViTAE-S | 224x224 | 82.0 | 95.9 | 87.0 | Coming Soon |

## Statement
This project is for research purpose only. For any other questions please contact [yufei.xu at outlook.com](mailto:yufei.xu@outlook.com) [qmzhangzz at hotmail.com](mailto:qmzhangzz@hotmail.com) .
