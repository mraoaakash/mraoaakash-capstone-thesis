#! /bin/bash
eval "$(conda shell.bash hook)"
conda activate pathldm

wandb disabled
python /home/aakashrao/research/research-thesis/mraoaakash-capstone-thesis/PathLDM/main.py -t --gpu 1  --base scripts/configs/clip_imagenet_finetune.yaml --name testing --train True --logdir /mnt/storage/aakashrao/cifsShare/PathLDM/outputs