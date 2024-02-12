import json
import os
import torch
import json
import uuid
import sys

from pathlib import Path
from transformers import pipeline
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from bert_score import score
from paraphrasing_model import QualityControlPipeline

basedir = Path(__file__).parents[1]
data_set_file_path = os.path.join(
    basedir, "Dataset/CoQA/Full_set/coqa.test.json")
result_set_file_path = os.path.join(
    os.path.dirname(__file__), "para_sets/coqa_score_set.json"
)
parameter_groups_file_path = os.path.join(
    os.path.dirname(__file__), "parameter_groups.json"
)


def get_rouge_scores(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)


def get_bleu_score(
    reference: str, candidate: str, smoothingFunction: SmoothingFunction
) -> str:
    reference = reference.split()
    candidate = candidate.split()
    return sentence_bleu(
        [reference], candidate, smoothing_function=smoothingFunction.method1
    )


def get_bert_score(reference: str, candidate: str) -> dict:
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}


# Initialize the paraphrasing model
model = QualityControlPipeline("sentences")
# Set the smoothing function for BLEU
chencherry = SmoothingFunction()

# Create the JSON structure
data = {"CoQA": {"ROUGE": [], "BLEU": [], "BERTScore": []}}

with open(
    data_set_file_path,
    "r",
) as file:
    dataset = json.load(file)

with open(parameter_groups_file_path, "r") as file:
    parameter_set = json.load(file)

for reference in dataset["data"]:
    for parameter in parameter_set:
        candidate = model(
            reference["story"],
            parameter_set[parameter]["lexical"],
            parameter_set[parameter]["syntactic"],
            parameter_set[parameter]["semantic"],
        )
        candidate = candidate[0]["generated_text"]
        # Fill the JSON structure
        for metric in ["ROUGE", "BLEU", "BERTScore"]:
            item = {
                "id": reference["id"],
                "reference_text": reference["story"],
                "candidate_text": candidate,
                "model_parameter_group": parameter,
                "metric": {},
            }
            if metric == "ROUGE":
                rouge_scores = get_rouge_scores(reference["story"], candidate)
                item["metric"] = {
                    "rouge1": {
                        "f1": rouge_scores["rouge1"].fmeasure,
                        "p": rouge_scores["rouge1"].precision,
                        "r": rouge_scores["rouge1"].recall,
                    },
                    "rouge2": {
                        "f1": rouge_scores["rouge2"].fmeasure,
                        "p": rouge_scores["rouge2"].precision,
                        "r": rouge_scores["rouge2"].recall,
                    },
                    "rougeL": {
                        "f1": rouge_scores["rougeL"].fmeasure,
                        "p": rouge_scores["rougeL"].precision,
                        "r": rouge_scores["rougeL"].recall,
                    },
                }
            elif metric == "BLEU":
                item["metric"] = {
                    "bleu": {
                        "score": get_bleu_score(
                            reference["story"], candidate, chencherry
                        )
                    }
                }
            elif metric == "BERTScore":
                bert_scores = get_bert_score(reference["story"], candidate)
                item["metric"] = {
                    "bertscore": {
                        "f1": bert_scores["f1"],
                        "p": bert_scores["precision"],
                        "r": bert_scores["recall"],
                    }
                }
            data["CoQA"][metric].append(item)


# Write the filled JSON structure to the file
with open(result_set_file_path, "w") as file:
    json.dump(data, file, indent=4)
