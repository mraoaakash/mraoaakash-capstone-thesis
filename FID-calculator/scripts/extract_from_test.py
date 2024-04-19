import io
import os
import h5py
import tqdm
import argparse
import numpy as np
from PIL import Image
import pathlib as Path
from torch.utils.data import Dataset


def get_random_crop(img, size):
    x = np.random.randint(0, img.shape[1] - size)
    y = np.random.randint(0, img.shape[0] - size)
    img = img[y : y + size, x : x + size]
    return img

def TCGADataset(data_dir, token_num, outdir, crop_size=256):

    os.makedirs(outdir, exist_ok=True)

    data_file = h5py.File(os.path.join(data_dir, "TCGA_BRCA_10x_448_tumor.hdf5"), "r")

    train = np.load(os.path.join(data_dir, f"train_test_brca_tumor_{token_num}/train.npz"), allow_pickle=True)
    test = np.load(os.path.join(data_dir, f"train_test_brca_tumor_{token_num}/test.npz"), allow_pickle=True)
    indices_train = train["indices"]
    indices_test = test["indices"]

    indices = indices_test

    print(data_file.keys())

    for idx in tqdm.tqdm(indices):
        
        tile = data_file["X"][idx]
        folder_name = data_file["folder_name"][idx].decode("utf-8")
        wsi = data_file["wsi"][idx].decode("utf-8")

        if os.path.join(outdir, f"{wsi}_{folder_name}.png") in os.listdir(outdir):
            continue

        tile = np.array(Image.open(io.BytesIO(tile)))

        image = (tile / 127.5 - 1.0)
        if crop_size:
            image = get_random_crop(image, crop_size)

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
        
        # convert to PIL Image
        image = Image.fromarray((image * 127.5 + 127.5).astype(np.uint8))
        
        # save image
        os.makedirs(outdir, exist_ok=True)
        image.save(os.path.join(outdir, f"{wsi}_{folder_name}.png"))

    
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, required=True)
    argparser.add_argument("--token_num", type=int, required=True)
    argparser.add_argument("--outdir", type=str, required=True)
    args = argparser.parse_args()

    TCGADataset(args.data_dir, args.token_num, args.outdir)