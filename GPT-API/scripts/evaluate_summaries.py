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

