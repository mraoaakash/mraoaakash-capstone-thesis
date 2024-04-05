#! /bin/bash
eval "$(conda shell.bash hook)"
conda activate pathldm

BASEPATH=/storage/aakash.rao_asp24/research/research-thesis/PathLDM

cd $BASEPATH

wandb disabled
python main.py -t --gpu 1  --base scripts/configs/clip_imagenet_finetune.yaml --name testing --train True --logdir /mnt/storage/aakashrao/cifsShare/PathLDM/outputs