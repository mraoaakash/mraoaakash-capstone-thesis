# PathLDM: Text conditioned Latent Diffusion Model for Histopathology

This code-base started off from that mentioned in [PathLDM: Text conditioned Latent Diffusion Model for Histopathology.](https://openaccess.thecvf.com/content/WACV2024/papers/Yellapragada_PathLDM_Text_Conditioned_Latent_Diffusion_Model_for_Histopathology_WACV_2024_paper.pdf) and builds heavily on both [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) and [cvlab-stonybrook/PathLDM](https://github.com/cvlab-stonybrook/PathLDM/tree/main)

## Requirements
To install python dependencies, use the following code.

```
conda env create -f environment.yaml
conda activate pathldm
```

This is the updated environment after fixing the runtime errors.


## Downloading + Organizing Data
All the data and models are available at [This Google Drive Link](https://drive.google.com/drive/folders/1NLU_WV--joDhghHUbKkpoaoL-Hu237QX?usp=sharing)
- [Summaries](https://drive.google.com/drive/folders/1-K_BVqr7x535vQ4cSIWJhM4zknrOMdGZ?usp=drive_link)
- [Models](https://drive.google.com/drive/folders/1nNOBIYgb0u9om5vSaEeUlSSSb0flL8mK?usp=drive_link)
- [VAE and U-net](https://drive.google.com/drive/folders/1_urgkNKIMmFoATiRtDwgjGuwjEt_fdvG?usp=drive_link)
- [Image](https://drive.google.com/drive/folders/1MPBsVjh7q57DzYJXSLF2wkKjssw3jEtF?usp=drive_link)

## Pretrained models

We provide the following trained models

| Token Length |  FID  | Link |
|:--------------------:|:--------------------:|:-------------------------:|
| 154   | 22.39    | [link](https://drive.google.com/drive/folders/1WCZwQWFj5C2Whr7ufKrtCDI0VlXwrVOB?usp=sharing)    |
| 50    | 21.51    | [link](https://drive.google.com/drive/folders/1q7PVd08REyRMthFmhPvdvanzND-No3Ry?usp=sharing)    |
| 35    | 21.11    | [link](https://drive.google.com/drive/folders/1PlPJ15pDaCfNOX32JJ0ZDX-VAE5ZBuiU?usp=sharing)    |
| 20    | 24.01    | [link](https://drive.google.com/drive/folders/15gx8cdfiwBZTlTC5dO7LQeXRrb3G8mx_?usp=sharing)    |

## Compared to other SOTA works
| Token Length |  FID  | 
|:--------------------:|:--------------------:|
| Moghadam et.al     |  105.81   |
| Medfusion          |  39.49    |
| Stable Diffusion   |  30.56    |
| PathLDM            |  22.39    |
| Our Best           |  21.11    |




## Training
Models can be trained using the ```train.sh``` or ```mphasis_train.sh``` after changing the relevant paths depending on a single or multi-GPU environment respectively. You must create a config file, depending on your data. An example of this can be seen [here](https://github.com/mraoaakash/mraoaakash-capstone-thesis/blob/main/PathLDM/scripts/configs/MPHASIS_clip_imagenet_finetune.yaml). 

## Evaluation
Evaluation can be carried out as described in the [FID-calculator](https://github.com/mraoaakash/mraoaakash-capstone-thesis/tree/main/FID-calculator) sub-directory of this repo.
