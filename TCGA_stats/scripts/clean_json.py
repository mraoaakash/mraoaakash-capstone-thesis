import json
import argparse
import numpy as np
import pandas as pd

key_array = []
def get_all_keys(object_test):
    if isinstance(object_test, dict):
        for key, value in object_test.items():
            key_array.append(key)
            get_all_keys(value)
    elif isinstance(object_test, list):
        for item in object_test:
            get_all_keys(item)

def make_df(data):
    # recursively flatten json
    def flatten_json(y):
        out = {}

        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '_')
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(y)

        return out
    
    # flatten json
    data_flat = [flatten_json(d) for d in data]

    # convert to dataframe
    df = pd.DataFrame(data_flat)
    return df

def cleaner(df):
    # drop columns with all NaN
    df = df.dropna(axis=1, how='all')
    # split column names with regex [a-z]_[01]
    cols = df.columns
    col_cpy = cols.copy()
    cols = [col.split(r'_[01]')[-1] for col in cols]
    print(cols)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--json", help="Path to json file")
    argparser.add_argument("--outfolder", help="Path to output folder")
    args = argparser.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    # get_all_keys(data)

    Cols_of_interest = ['ajcc_pathologic_stage', 'case_id' , 'classification_of_tumor', 'ethnicity'  , 'gender', 'primary_diagnosis' , 'race', 'state' 'submitter_id', 'tissue_or_organ_of_origin' , 'treatment_or_therapy' , 'year_of_birth' , 'year_of_death', 'year_of_diagnosis']
    df = make_df(data)
    clean_df = cleaner(df)