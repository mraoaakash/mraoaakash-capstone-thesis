import os
import json
import argparse
import numpy as np

def npzer(input, output, token_len, summary):
    output = f'{output}/train_test_brca_tumor_{token_len}'
    os.makedirs(output, exist_ok=True)

    summary = f'{summary}/summaries_{token_len}/summaries_{token_len}.json'
    summary = json.load(open(summary))
    print(summary.keys())

    for i in ['train', 'test']:
        data = np.load(f'{input}/{i}.npz', allow_pickle=True)
        print(data.files)
        # ['indices', 'summaries', 'prob_tumor', 'prob_til', 'indices_tumor', 'indices_til']
        summaries = data['summaries'].tolist()
        prob_tumor = data['prob_tumor'].tolist()
        prob_til = data['prob_til'].tolist()
        indices = data['indices'].tolist()
        # indices_tumor = data['indices_tumor']
        # indices_til = data['indices_til']
        # np.savez(f'{output}/{i}.npz', indices=indices, summaries=summaries, prob_tumor=prob_tumor, prob_til=prob_til)
        break









if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str)
    argparser.add_argument('--summary', type=str)
    argparser.add_argument('--output', type=str)
    argparser.add_argument('--token_len', type=int)
    args = argparser.parse_args()

    npzer(args.input, args.output, args.token_len, args.summary)