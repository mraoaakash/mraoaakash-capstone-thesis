#! /bin/bash
eval "$(conda shell.bash hook)"
conda activate pathldm

BASEPATH=/home/aakashrao/research/research-thesis/mraoaakash-capstone-thesis/PathLDM

cd $BASEPATH

wandb disabled
# python main.py -t --gpus 1  --base scripts/configs/MPHASIS_clip_imagenet_finetune.yaml --name testing --train True --logdir /mnt/storage/aakashrao/cifsShare/PathLDM/outputs

python main.py -t --gpus 1  --base scripts/configs/MPHASIS_clip_imagenet_finetune.yaml --name testing --train True --logdir /mnt/storage/aakashrao/cifsShare/PathLDM/outputs/test_memory