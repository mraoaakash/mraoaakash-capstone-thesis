import argparse
import shutil
import os
import tqdm
import numpy as np
import random
import cv2

def make_uniform(reference_dir, norm_dir, outdir):
    ref_images = np.array(os.listdir(reference_dir))
    norm_images = np.array(os.listdir(norm_dir))

    intersection = np.intersect1d(ref_images, norm_images)
    print(f'Intersection: {len(intersection)}')
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        

    pass

if '__name__' == '__main__':
    parser = argparse.ArgumentParser(description='Make a dataset uniform')
    parser.add_argument('--reference_dir', type=str, required=True, help='Path to the reference dataset')
    parser.add_argument('--norm_dir', type=str, required=True, help='Path to the dataset to be normalized')
    parser.add_argument('--outdir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()

    make_uniform(args.reference_dir, args.norm_dir, args.outdir)