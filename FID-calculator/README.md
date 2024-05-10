# FID-calculator
This sub-directory is used for the standardised process of synthetic data generation, as well as model evaluation.

The environment for this subdir is the same as that of PathLDM, and can be found in the PathLDM sub-directory. Please ensure you have installed all the required packages before attempting to recreeate this work.

## Synthetic data generation
This process is carried out using DDIM sampling, as described in the original [PathLDM repo](https://github.com/cvlab-stonybrook/PathLDM/tree/main) as well as the [latent-diffusion repo](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file). The ```run.sh``` file provided in this subdir contains the main examples to run the generation workflow and involves the following code:

### Generate the original images from their .npz file
```
  echo "Processing for token length $i"
  python scripts/extract_from_test.py \
      --data_dir # base path to where all your summaries are stored \
      --token_num  # token number of choice   \
      --outdir output directory of choice

```

### Generate images using different models
```
python scripts/generate_images.py \
    --ckpt_path # Model file path .pt file  \
    --config_path # path to the config file \
    --data_dir # path to the summary folder \
    --token_num # token number \
    --batch_size # batch size \
    --outdir # directory to save generated images
```

## Model evaluation
This process is carried out using the PyTorch-FID package and can be done as follows. First you generate the original data statistics to save time and not have to constantly recompute itfor every evaluation step. This can be done using:
```
python -m pytorch_fid --save-stats inputFolder outputFile
```

Subsiquently, you can calculate the model evaluation score using the following
```
python -m pytorch_fid originalImageFolder syntheticDataFolder --batch-size 128
```
