#! /bin/bash
#PBS -N PathLDM_Trainer
#PBS -o PathLDM_Trainer_out.log
#PBS -e PathLDM_Trainer_err.log
#PBS -l ncpus=50
#PBS -q gpu
#PBS -l host=compute3

eval "$(conda shell.bash hook)"
conda activate ldm

BASEPATH=/storage/aakash.rao_asp24/research/research-thesis/PathLDM

cd $BASEPATH

wandb disabled
python main.py -t  --base scripts/configs/clip_imagenet_finetune.yaml 