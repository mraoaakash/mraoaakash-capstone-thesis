import evaluate
import os
import json
import argparse
import numpy as np
import pandas as pd


def BLEU_evaluation(merged, token_num):
    # 'bleu': bleu score,
    # 'precisions': geometric mean of n-gram precisions,
    # 'brevity_penalty': brevity penalty,
    # 'length_ratio': ratio of lengths,
    # 'translation_length': translation_length,
    # 'reference_length': reference_length

    bleu = evaluate.load("bleu")
    bleu_scores = pd.DataFrame(columns=["id", "bleu", "precisions", "brevity_penalty", "length_ratio", "translation_length", "reference_length"])
    bleu_scores["id"] = merged["id"]
    for index, row in merged.iterrows():
        bleu = bleu.compute(predictions=[row[f"summary_{token_num}"]], references=[row["summary_long"]])
        print(f"BLEU score for {row['id']} is {bleu}")
        id_of_row = row["id"]
        bleu_scores.loc[bleu_scores["id"] == id_of_row, "bleu"] = bleu["bleu"]
        bleu_scores.loc[bleu_scores["id"] == id_of_row, "precisions"] = bleu["precisions"]
        bleu_scores.loc[bleu_scores["id"] == id_of_row, "brevity_penalty"] = bleu["brevity_penalty"]
        bleu_scores.loc[bleu_scores["id"] == id_of_row, "length_ratio"] = bleu["length_ratio"]
        bleu_scores.loc[bleu_scores["id"] == id_of_row, "translation_length"] = bleu["translation_length"]
        bleu_scores.loc[bleu_scores["id"] == id_of_row, "reference_length"] = bleu["reference_length"]
        

        break

    print(bleu_scores.head())


if __name__ =="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--token_num", type=str, required=True)
    args = argparser.parse_args()

    main_summary_path = os.path.join(args.input, "all_cleaned_summaries.csv")
    other_summary_path = os.path.join(args.input, f"summaries_{args.token_num}", f"summaries_{args.token_num}.json")

    main_summaries = pd.read_csv(main_summary_path)
    main_summaries = main_summaries[["submitter_slide_ids","summary_long"]]
    main_summaries = main_summaries.rename(columns={"submitter_slide_ids":"id"})

    
    other_summaries = json.load(open(other_summary_path, "r"))
    other_summaries_df = pd.DataFrame(columns=["id", f"summary_{args.token_num}"])
    other_summaries_df["id"] = [key for key in other_summaries.keys()]
    other_summaries_df[f"summary_{args.token_num}"] = [value for value in other_summaries.values()]

    merged = pd.merge(main_summaries, other_summaries_df, on="id")

    bleu_scores = BLEU_evaluation(merged, args.token_num)