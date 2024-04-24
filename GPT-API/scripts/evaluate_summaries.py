import evaluate
import os
import json
import argparse
import numpy as np
import pandas as pd


def BLEU_evaluation(referenes, predictions):
    bleu_scores = []
    bleu = evaluate.load("bleu")
    for i in range(len(referenes)):
        reference = referenes[i]
        prediction = predictions[i]
        bleu_score = bleu.compute(reference, prediction)
        bleu_scores.append(bleu_score)
    return bleu_scores


if __name__ =="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--token_num", type=str, required=True)
    args = argparser.parse_args()

    main_summary_path = os.path.join(args.input, "all_cleaned_summaries.csv")
    other_summary_path = os.path.join(args.input, f"summaries_{args.token_num}", f"summaries_{args.token_num}.json")

    main_summaries = pd.read_csv(main_summary_path)
    
    other_summaries = json.load(open(other_summary_path, "r"))
    other_summaries = pd.DataFrame(other_summaries)

