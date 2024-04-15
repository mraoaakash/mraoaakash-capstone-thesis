import os 
import argparse
import json
import numpy as np
import pandas as pd


def clean_data(data):
    interest = ["submitter_id", "case_id", "submitter_slide_ids"]

    main_items = np.array([]) 
    i=0
    for item in data:
        temp = {}
        if i ==0:
            i+=1
        for key in interest:
            temp[key] = item[key]
        main_items = np.append(main_items, temp)


    df = pd.DataFrame(columns=interest)
    for item in main_items:
        for i in range(len(item['submitter_slide_ids'])):
            temp = {}
            temp['submitter_id'] = item['submitter_id']
            temp['case_id'] = item['case_id']
            temp['submitter_slide_ids'] = item['submitter_slide_ids'][i]
            df = pd.concat([df, pd.DataFrame(temp, index=[0])], ignore_index=True)

    return df


def get_data(input, summaryfolder, master, outputfolder):
    master = pd.read_csv(master)
    train_set = json.load(open(os.path.join(summaryfolder, 'summaries_list_train.json')))
    train_keys = list(train_set.keys())
    test_set = json.load(open(os.path.join(summaryfolder, 'summaries_list_test.json')))
    test_keys = list(test_set.keys())
    allkeys = train_keys + test_keys
    all_summary = {**train_set, **test_set}

    input = json.load(open(input))
    input = clean_data(input)
    input["summary_75"] = np.nan
    master['patient_filename'] = master['patient_filename'].apply(lambda x: x.split('.')[0])
    common = []
    for key in allkeys:
        if key in input['submitter_slide_ids'].tolist():
            common.append(key)
            # find location of key in input and add summary_75
            index = input.index[input['submitter_slide_ids'] == key].tolist()[0]
            input.at[index, 'summary_75'] = all_summary[key]

    
    # dropping all rows with nan values
    input = input.dropna()
    input = input.reset_index(drop=True)
    
    master = master.set_index('patient_filename').T.to_dict('list')
    input["summary_long"]  = np.nan


    for i in range(len(input)):
        submitter_id = input['submitter_id'][i]
        if submitter_id in master.keys():
            input.at[i, 'summary_long'] = master[submitter_id][0]

        else:
            input.at[i, 'summary_long'] = input['summary_75'][i]

    print(input.iloc[0]['summary_long'])    
    input.to_csv(os.path.join(outputfolder, 'all_cleaned_summaries.csv'), index=False)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--summaryfolder', type=str, default='summary')
    argparser.add_argument('--master', type=str, default='master.csv')
    argparser.add_argument('--input', type=str, default='input.json')
    argparser.add_argument('--outputfolder', type=str, default='output')
    args = argparser.parse_args()

    get_data(args.input, args.summaryfolder, args.master, args.outputfolder)