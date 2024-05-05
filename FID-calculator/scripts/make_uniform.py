import argparse
import shutil
import os
import tqdm
import numpy as np
import random
import cv2

def intersection_gen(reference_dir, norm_dir, outdir):
    print("running")
    ref_images = np.array(os.listdir(reference_dir))
    norm_images = np.array(os.listdir(norm_dir))

    intersection = np.intersect1d(ref_images, norm_images)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        

    pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Make a dataset uniform')
    argparser.add_argument('--reference_dir', type=str, required=True, help='Path to the reference dataset')
    argparser.add_argument('--norm_dir', type=str, required=True, help='Path to the dataset to be normalized')
    argparser.add_argument('--outdir', type=str, required=True, help='Path to the output directory')
    args = argparser.parse_args()

    print(args)


    intersection_gen(args.reference_dir, args.norm_dir, args.outdir)