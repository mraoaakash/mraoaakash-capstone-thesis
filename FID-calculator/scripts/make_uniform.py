import argparse
import shutil
import os
import tqdm
import numpy as np
import random
import cv2


INTERSECTION = []
def intersection_gen(reference_dir, norm_dir, outdir):
    print("running")
    ref_images = np.array(os.listdir(reference_dir))
    norm_images = np.array(os.listdir(norm_dir))

    return np.intersect1d(ref_images, norm_images)

def maker(token, basedir, outdir, intersect):
    basedir = os.path.join(basedir, f"generated_{token}", "images")
    outdir = os.path.join(outdir, f"gen_{token}", "images")
    print(f"Making dataset uniform for {token}")
    os.makedirs(outdir, exist_ok=True)

    # print(intersect)
    # print(os.listdir(basedir))

    for img in tqdm.tqdm(intersect):
        shutil.copy(os.path.join(basedir, img), os.path.join(outdir, img))
        break

    pass
    
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Make a dataset uniform')
    argparser.add_argument('--reference_dir1', type=str, required=True, help='Path to the reference dataset')
    argparser.add_argument('--reference_dir2', type=str, required=True, help='Path to the reference dataset')
    argparser.add_argument('--norm_dir', type=str, required=True, help='Path to the dataset to be normalized')
    argparser.add_argument('--outdir', type=str, required=True, help='Path to the output directory')
    args = argparser.parse_args()

    INTERSECTION = intersection_gen(args.reference_dir1, args.reference_dir2, args.outdir)
    print(INTERSECTION)

    for token in ["20"]:
        maker(token, args.norm_dir, args.outdir, INTERSECTION)

