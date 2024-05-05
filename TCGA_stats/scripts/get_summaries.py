import argparse
import os
import pandas as pd
import numpy as np
import json

def generate_10_files(data_dir):
    path = os.path.join(data_dir, "summaries_75", "summaries_list_test.json")
    data = ''
    with open(path) as f:
        data = json.load(f)
    summaries = np.array(list(data.keys()))
    # get 10 random files
    np.random.seed(19)
    file_arr = np.random.choice(summaries, 10, replace=False)

    for file in file_arr:
        print(file)

    return file_arr

def get_summaries(data_dir, token_num,file_arr):
    folder = os.path.join(data_dir, "summaries_txts", f"summaries_{token_num}")
    for file in file_arr:
        file_path = os.path.join(folder, f"{file}.txt")
        # read the file
        try:
            with open(file_path) as f:
                lines = f.readlines()
                print(lines[0])
        except:
            print(f"Error reading file {file_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--token_num", type=int, required=True)
    args = parser.parse_args()
    file_arr = generate_10_files(args.data_dir)

    get_summaries(args.data_dir, args.token_num, file_arr)
