import os
import json
import evaluate
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def BLEU_evaluation(merged, token_num):
    # 'bleu': bleu score,
    # 'precisions': geometric mean of n-gram precisions,
    # 'brevity_penalty': brevity penalty,
    # 'length_ratio': ratio of lengths,
    # 'translation_length': translation_length,
    # 'reference_length': reference_length

    bleu_model = evaluate.load("bleu")
    bleu_scores = pd.DataFrame(columns=["id", "bleu", "precisions", "brevity_penalty", "length_ratio", "translation_length", "reference_length"])
    for index, row in tqdm(merged.iterrows(), total=merged.shape[0]):
        bleu = bleu_model.compute(predictions=[row[f"summary_{token_num}"]] if isinstance(row[f"summary_{token_num}"], str) else ["Not Given"], references=[row["summary_long"] if isinstance(row["summary_long"], str) else ["Not Given"]])
        # print(f"BLEU score for {row['id']} is {bleu}")
        df_temp = pd.DataFrame({
            "id": [row["id"]],
            "bleu": [bleu["bleu"]],
            "precisions": [bleu["precisions"]],
            "brevity_penalty": [bleu["brevity_penalty"]],
            "length_ratio": [bleu["length_ratio"]],
            "translation_length": [bleu["translation_length"]],
            "reference_length": [bleu["reference_length"]]
        })
        bleu_scores = pd.concat([bleu_scores, df_temp])

        

        # break

    # print(bleu_scores.head())

    # print statistics of bleu scores
    print(f"Mean BLEU score: {np.mean(bleu_scores['bleu'])}")
    print(f'Median BLEU score: {np.median(bleu_scores["bleu"])}')
    print(f'Min BLEU score: {np.min(bleu_scores["bleu"])}')
    print(f'Max BLEU score: {np.max(bleu_scores["bleu"])}')
    print(f'Std BLEU score: {np.std(bleu_scores["bleu"])}')
    print(f'Q1 BLEU score: {np.percentile(bleu_scores["bleu"], 25)}')
    print(f'Q3 BLEU score: {np.percentile(bleu_scores["bleu"], 75)}')



def ROUGE_evaluation(merged, token_num):
    # 'bleu': bleu score,
    # 'precisions': geometric mean of n-gram precisions,
    # 'brevity_penalty': brevity penalty,
    # 'length_ratio': ratio of lengths,
    # 'translation_length': translation_length,
    # 'reference_length': reference_length

    rouge_model = evaluate.load("rouge")
    rouge_scores = pd.DataFrame(columns=["id", 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    for index, row in tqdm(merged.iterrows(), total=merged.shape[0]):
        rouge = rouge_model.compute(predictions=[row[f"summary_{token_num}"]] if isinstance(row[f"summary_{token_num}"], str) else ["Not Given"], references=[row["summary_long"] if isinstance(row["summary_long"], str) else ["Not Given"]])
        # print(f"rouge score for {row['id']} is {rouge}")
        df_temp = pd.DataFrame({
            "id": [row["id"]],
            'rouge1': [rouge["rouge1"]],
            'rouge2': [rouge["rouge2"]],
            'rougeL': [rouge["rougeL"]],
            'rougeLsum': [rouge["rougeLsum"]]
        })
        rouge_scores = pd.concat([rouge_scores, df_temp])
    print(f"Mean rouge1 score: {np.mean(rouge_scores['rouge1'])}")
    print(f"Median rouge1 score: {np.median(rouge_scores['rouge1'])}")
    print(f"Min rouge1 score: {np.min(rouge_scores['rouge1'])}")
    print(f"Max rouge1 score: {np.max(rouge_scores['rouge1'])}")
    print(f"Std rouge1 score: {np.std(rouge_scores['rouge1'])}")
    print(f"Q1 rouge1 score: {np.percentile(rouge_scores['rouge1'], 25)}")
    print(f"Q3 rouge1 score: {np.percentile(rouge_scores['rouge1'], 75)}")

    return




if __name__ =="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--token_num", type=str, required=True)
    args = argparser.parse_args()


    print("-----------------Evaluating Summaries-----------------")
    print(f"Input Path: {args.input}")
    print(f"Output Path: {args.output}")
    print(f"Token Number: {args.token_num}")
    print("------------------------------------------------------")

    main_summary_path = os.path.join(args.input, "all_cleaned_summaries.csv")
    if args.token_num !=75:
        other_summary_path = os.path.join(args.input, f"summaries_{args.token_num}", f"summaries_{args.token_num}.json")

        main_summaries = pd.read_csv(main_summary_path)
        main_summaries = main_summaries[["submitter_slide_ids","summary_long"]]
        main_summaries = main_summaries.rename(columns={"submitter_slide_ids":"id"})

        
        other_summaries = json.load(open(other_summary_path, "r"))
        other_summaries_df = pd.DataFrame(columns=["id", f"summary_{args.token_num}"])
        other_summaries_df["id"] = [key for key in other_summaries.keys()]
        other_summaries_df[f"summary_{args.token_num}"] = [value for value in other_summaries.values()]
        merged = pd.merge(main_summaries, other_summaries_df, on="id")

    else:
        merged = pd.read_csv(main_summary_path)
        merged = merged[["submitter_slide_ids","summary_long","summary_75"]]
        merged = merged.rename(columns={"submitter_slide_ids":"id"})
        merged = merged.rename(columns={"summary_75":"summary_75"})
        



    # bleu_scores = BLEU_evaluation(merged, args.token_num)
    rogue_scores = ROUGE_evaluation(merged, args.token_num)

    print("------------------------------------------------------")
    print("Evaluation Completed")
    print("------------------------------------------------------")
    print("\n\n\n\n\n")