import os
import json
import argparse
import numpy as np

def npzer(input, output, token_len, summary):
    output = f'{output}/train_test_brca_tumor_{token_len}'
    os.makedirs(output, exist_ok=True)

    summary = f'{summary}/summaries_{token_len}/summaries_{token_len}.json'
    summary = json.load(open(summary))

    for i in ['train', 'test']:
        data = np.load(f'{input}/{i}.npz', allow_pickle=True)
        # ['indices', 'summaries', 'prob_tumor', 'prob_til', 'indices_tumor', 'indices_til']
        prob_tumor = data['prob_tumor'].tolist()
        prob_til = data['prob_til'].tolist()
        indices = data['indices'].tolist()

        # make summary into its correct type of dictionary
        summaries = data['summaries'].tolist()
        for key in summaries.keys():
            try:
                summaries[key] = summary[key]
            except:
                print(f'{key} not found')
                summaries[key] = summaries[key][:token_len]
        # indices_tumor = data['indices_tumor']
        # indices_til = data['indices_til']
        np.savez(f'{output}/{i}.npz', indices=indices, summaries=summaries, prob_tumor=prob_tumor, prob_til=prob_til)









if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str)
    argparser.add_argument('--summary', type=str)
    argparser.add_argument('--output', type=str)
    argparser.add_argument('--token_len', type=int)
    args = argparser.parse_args()

    npzer(args.input, args.output, args.token_len, args.summary)