import warnings
from pathlib import Path
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from torchvision import transforms

from PathLDM.ddim import DDIMSampler
from PathLDM.ldm_utils import instantiate_from_config
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from einops import rearrange
import pickle
import sys
import pandas as pd
import h5py
import io
import os
import tqdm


def TCGADataset(data_dir, token_num, outdir, crop_size=256):

    os.makedirs(outdir, exist_ok=True)

    data_file = h5py.File(os.path.join(data_dir, "TCGA_BRCA_10x_448_tumor.hdf5"), "r")

    train = np.load(os.path.join(data_dir, "TCGA_Dataset", f"train_test_brca_tumor_{token_num}/train.npz"), allow_pickle=True)
    test = np.load(os.path.join(data_dir, "TCGA_Dataset", f"train_test_brca_tumor_{token_num}/test.npz"), allow_pickle=True)
    # indices_train = train["indices"]
    indices_test = test["indices"]
    summaries = test["summaries"].tolist()
    prob_tumor = test["prob_tumor"].tolist()
    prob_til = test["prob_til"].tolist()




    # indices = np.concatenate([indices_train, indices_test])
    levels = ["low", "high"]
    num_levels = 2

    captions = pd.DataFrame(columns=["idx", "caption"])

    for idx in tqdm.tqdm(indices_test):

        wsi_name = data_file["wsi"][idx].decode()
        folder_name = data_file["folder_name"][idx].decode()
        text_prompt = summaries[wsi_name]

        text_name = f"{wsi_name}_{folder_name}"

        text_prompt = summaries[wsi_name]

        # Convert tumor infiltrating lymphocytes to levels low / mid / high and add to text prompt
        p_til = prob_til.get(wsi_name, {}).get(folder_name)
        if p_til is not None:
            p_til = int(p_til * len(levels))
            text_prompt = f"{levels[p_til]} til; {text_prompt}"

        # Convert tumor presence to levels low / mid / high and add to text prompt
        p_tumor = prob_tumor.get(wsi_name, {}).get(folder_name)
        if p_tumor is not None:
            p_tumor = int(p_tumor * len(levels))
            text_prompt = f"{levels[p_tumor]} tumor; {text_prompt}"

        # Replace text prompt with unconditional text prompt with probability p_uncond
        # Dont replace if p_til is positive
        if np.random.rand() < 0.1 and (p_til is None or p_til == 0):
            text_prompt = ""

        temp_df = pd.DataFrame({"idx": [text_name], "caption": [text_prompt]})
        captions = pd.concat([captions, temp_df])
        
    return captions


device = torch.device('cuda:0')
def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def get_model(config_path,  device, checkpoint):
    config = OmegaConf.load(config_path)
    del config['model']['params']['first_stage_config']['params']['ckpt_path']
    del config['model']['params']['unet_config']['params']['ckpt_path']
    model = load_model_from_config(config, checkpoint, device)
    return model

def get_conditional_token(batch_size, summary):
    # append tumor and TIL probability to the summary
    tumor = ["High tumor; low TIL;"]*(batch_size)
    return [t+summary for t in tumor]

def get_unconditional_token(batch_size):
        return [""]*batch_size

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ckpt_path", type=str, required=True)
    argparser.add_argument("--config_path", type=str, required=True)
    argparser.add_argument("--data_dir", type=str, required=True)
    argparser.add_argument("--token_num", type=str, required=True)
    argparser.add_argument("--outdir", type=str, required=True)
    args = argparser.parse_args()
    

    model = get_model(args.config_path, device, args.ckpt_path)
    sampler = DDIMSampler(model)

    if not os.path.exists(os.path.join(args.outdir, "summaries.csv")):
        summaries = TCGADataset(args.data_dir, args.token_num, args.outdir)
        print(summaries)
        summaries.to_csv(os.path.join(args.outdir, "summaries.csv"), index=False)
    else:
        summaries = pd.read_csv(os.path.join(args.outdir, "summaries.csv"))
    
    batch_size = 16
    shape = [3,64,64]

    scale = 1.5

    outdir = os.path.join(args.outdir, "images")
    os.makedirs(outdir, exist_ok=True)

    # make np arrays of ids and summaries and reshape to batch_size
    ids = np.array(summaries["idx"])
    summaries = np.array(summaries["caption"])

    # reshape with batch_size and add padding to last batch
    ids = np.pad(ids, (0, batch_size - len(ids) % batch_size), mode='constant', constant_values=0)
    ids = ids.reshape(-1, batch_size)

    summaries = np.pad(summaries, (0, batch_size - len(summaries) % batch_size), mode='constant', constant_values="")
    summaries = summaries.reshape(-1, batch_size)
    # convert each element in summaries to string
    summaries = [[str(s) for s in summary] for summary in summaries]
    
    for summary, file in zip(summaries, ids):
        # convert summary to string
        with torch.no_grad():
            #unconditional token for classifier free guidance
            ut = get_unconditional_token(batch_size)
            uc = model.get_learned_conditioning(ut)
            
            # ct = get_conditional_token(batch_size, summary)
            cc = model.get_learned_conditioning(summary)
            
            samples_ddim, _ = sampler.sample(50, batch_size, shape, cc, verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=0)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()

            for i, sample in enumerate(x_samples_ddim):
                sample = rearrange(sample, 'c h w -> h w c')
                sample = Image.fromarray(sample.numpy())
                sample.save(os.path.join(outdir, f"{file[i]}.png"))

            break