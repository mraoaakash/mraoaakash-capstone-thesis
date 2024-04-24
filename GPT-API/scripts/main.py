import os
import json
import argparse
import numpy as np
import pandas as pd
from infer import get_encoding_k_summary, get_response, get_encoding_keywords

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('API key is missing')


def infer(input, token_len,outpath, type='k_summary'):
    if type == 'k_summary':
        input = get_encoding_k_summary(input, token_len)
    else:
        input = get_encoding_keywords(input, token_len)

    response = get_response(input, token_len)
    print(input)
    print(response)
    with open(outpath, 'w+') as f:
        f.write(response)
    return 





def main(input, output, token_len, type='k_summary'):
    if type == 'k_summary':
        output = os.path.join(output, f'summaries_txts', f'summaries_{token_len}')
    else:
        output = os.path.join(output, f'summaries_txts', f'summaries_keywords')
    os.makedirs(output, exist_ok=True)


    input = pd.read_csv(os.path.join(input, 'all_cleaned_summaries.csv'))
    inp_cpy = input[["submitter_slide_ids", "summary_long"]]
    print(inp_cpy.head())

    for index, row in inp_cpy.iterrows():
        output_path = os.path.join(output, f'{row["submitter_slide_ids"]}.txt')
        if os.path.exists(output_path):
            continue
        else:
            infer(row["summary_long"], token_len, output_path)
        # break

def generate_json(folderpath, outputpath, type='k_summary'):
    print(folderpath)
    print(outputpath)
    os.makedirs(os.path.dirname(outputpath), exist_ok=True)
    if type == 'k_summary':
        outputpath  = os.path.join(outputpath, f'summaries_{args.token_len}.json')
    else:
        outputpath  = os.path.join(outputpath, f'summaries_keywords.json')

    files = os.listdir(folderpath)

    try:
        files.remove('.DS_Store')
    except:
        pass
    
    # sort files
    files = sorted(files)

    data = {}

    for file in files:  
        file_name = file.split('.')[0]  
        file_path = os.path.join(folderpath, file)
        content = None
        with open(file_path, 'r') as f:
            content = f.read()
        data[file_name] = content

    with open(outputpath, 'w+') as f:
        json.dump(data, f)
        



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    argparser.add_argument('--token_len', type=int, required=True)
    argparser.add_argument('--type', type=str, default='k_summary')
    args = argparser.parse_args()

    main(args.input, args.output, args.token_len, args.type)
    # generate_json(os.path.join(args.output, "summaries_txts", f'summaries_{args.token_len}'), os.path.join(args.output, f'summaries_{args.token_len}' if args.type == 'k_summary' else 'summaries_keywords'), args.type)

