import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    cols_interest = ['diagnoses_0_ajcc_pathologic_stage','diagnoses_0_classification_of_tumor','demographic_ethnicity','demographic_gender','diagnoses_0_primary_diagnosis','demographic_race','demographic_state','diagnoses_0_tissue_or_organ_of_origin','diagnoses_0_treatments_0_treatment_or_therapy','demographic_year_of_birth','demographic_year_of_death','diagnoses_0_year_of_diagnosis','diagnoses_0_treatments_1_treatment_or_therapy']
    df = df[cols_interest]
    df['treatment_or_therapy'] = df['diagnoses_0_treatments_1_treatment_or_therapy'].combine_first(df['diagnoses_0_treatments_0_treatment_or_therapy'])
    df = df.drop(columns=['diagnoses_0_treatments_0_treatment_or_therapy','diagnoses_0_treatments_1_treatment_or_therapy'])
    rename = ['stage','tumor_classification','ethnicity','gender','primary_diagnosis','race','state','origin','treatment','birth','death','year_of_diagnosis']
    df.columns = rename
    print(df.columns)
    return df


def plotter(df):

    # figure with three subfigures
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    cols_of_interest=['ethnicity','gender','race']
    # df = df[cols_of_interest]
    df['ethnicity'] = [x if x in ['not hispanic or latino', 'hispanic or latino'] else 'other' for x in df["ethnicity"]]    
    df['ethnicity'] = ["Non-hispanic\nNon-latino" if x in ['not hispanic or latino'] else x for x in df["ethnicity"]]    
    df['ethnicity'] = ["Lispanic\nLatino" if x in ['hispanic or latino'] else x for x in df["ethnicity"]]    
    
    
    df['race'] = ["Other\n   " if x in ['not reported'] else x for x in df["race"]]    
    df['race'] = ["White\n   " if x in ['white'] else x for x in df["race"]]    
    df['race'] = ["Black\n   " if x in ['black or african american'] else x for x in df["race"]]    
    df['race'] = ["Asian\n   " if x in ['asian'] else x for x in df["race"]]    
    df['race'] = ["Native\n   " if x in ['american indian or alaska native'] else x for x in df["race"]]  
    
    
    df['gender'] = ["Male\n   " if x in ['male'] else x for x in df["gender"]]  
    df['gender'] = ["Female\n   " if x in ['female'] else x for x in df["gender"]]  
    df['gender'] = ["Other\n   " if x in ['other'] else x for x in df["gender"]]  



    

    for i, col in enumerate(cols_of_interest):
        df[col].value_counts().plot(kind='bar', ax=ax[i], color='#4893AF', edgecolor='black', linewidth=1)
        for p in ax[i].patches:
            ax[i].annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        ax[i].set_title(col.title(), fontsize=12, fontweight='bold')
        ax[i].set_ylim(0, 1201)
        ax[i].set_yticks(np.arange(0, 1201, 200))
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0)
        ax[i].set_xlabel(col.title(), fontsize=10)
        ax[i].set_ylabel('Count', fontsize=10)
    
    plt.suptitle("Demographic Information in TCGA-BRCA", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{args.outfolder}/demographic.png", dpi=300)
    # plt.show()
    plt.close()


    pds = np.array(df['primary_diagnosis']).astype(str)
    pds = np.unique(pds)
    # for ps in pds:
    #     print(ps)
    
    df['primary_diagnosis'] = ["IDC\n   " if x in ['Infiltrating duct and lobular carcinoma'] else x for x in df["primary_diagnosis"]]  
    df['primary_diagnosis'] = ["IDC\n   " if x in ['Infiltrating duct carcinoma, NOS'] else x for x in df["primary_diagnosis"]]   
    df['primary_diagnosis'] = ["IDC\n   " if x in ['Infiltrating duct mixed with other types of carcinoma'] else x for x in df["primary_diagnosis"]]   

    df['primary_diagnosis'] = ["ILC\n   " if x in ['Infiltrating lobular mixed with other types of carcinoma'] else x for x in df["primary_diagnosis"]]   
    df['primary_diagnosis'] = ["ILC\n   " if x in ['Lobular carcinoma, NOS'] else x for x in df["primary_diagnosis"]]   

    df['primary_diagnosis'] = [x if x in ['IDC\n   ', 'ILC\n   '] else "Other\n   " for x in df["primary_diagnosis"]]


    plt.figure(figsize=(5, 5))
    df['primary_diagnosis'].value_counts().plot(kind='bar', color='#4893AF', edgecolor='black', linewidth=1)
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.title('Primary Diagnosis', fontsize=16, fontweight='bold')
    plt.xlabel('Primary Diagnosis', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.ylim(0, 1201)
    plt.yticks(np.arange(0, 1201, 200))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{args.outfolder}/primary_diagnosis.png", dpi=300)
    # plt.show()
    plt.close()





    
    pass


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
    print(clean_df.head())

    plotter(clean_df)
    