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
    basedir, "dataset/Emotianal_Understanding/COM2SENSE/COM2SENSE.json"
)
result_set_file_path = os.path.join(
    os.path.dirname(__file__), "para_sets/com2sense_score_set.json"
)
parameter_groups_file_path = os.path.join(
    os.path.dirname(__file__), "parameter_groups.json"
)


def get_rouge_scores(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)


def get_bleu_score(
    reference: str, candidate: str, smoothingFunction: SmoothingFunction
) -> str:
    reference = reference.split()
    candidate = candidate.split()
    return str(
        sentence_bleu(
            [reference], candidate, smoothing_function=smoothingFunction.method1
        )
    )


def get_bert_score(reference: str, candidate: str) -> dict:
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}


# Initialize the paraphrasing model
model = QualityControlPipeline("sentences")
# Set the smoothing function for BLEU
chencherry = SmoothingFunction()

# Create the JSON structure
data = {"COM2SENSE": {"ROUGE": [], "BLEU": [], "BERTScore": []}}

with open(
    data_set_file_path,
    "r",
) as file:
    dataset = json.load(file)

with open(parameter_groups_file_path, "r") as file:
    parameter_set = json.load(file)

for reference in dataset["examples"]:
    for parameter in parameter_set:
        print(f'Processing reference: {reference["sent"]}')
        candidate = model(
            reference["sent"],
            parameter_set[parameter]["lexical"],
            parameter_set[parameter]["syntactic"],
            parameter_set[parameter]["semantic"],
        )
        candidate = candidate[0]["generated_text"]
        # Fill the JSON structure
        for metric in ["ROUGE", "BLEU", "BERTScore"]:
            print(f"Processing metric: {metric}")
            item = {
                "id": reference["id"],
                "reference_text": reference["sent"],
                "candidate_text": candidate,
                "model_parameter_group": parameter,
                "metric": {},
            }
            if metric == "ROUGE":
                item["metric"] = {
                    "rouge1": {
                        "f1": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rouge1"
                            ].fmeasure
                        ),
                        "p": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rouge1"
                            ].precision
                        ),
                        "r": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rouge1"
                            ].recall
                        ),
                    },
                    "rouge2": {
                        "f1": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rouge2"
                            ].fmeasure
                        ),
                        "p": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rouge2"
                            ].precision
                        ),
                        "r": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rouge2"
                            ].recall
                        ),
                    },
                    "rougeL": {
                        "f1": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rougeL"
                            ].fmeasure
                        ),
                        "p": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rougeL"
                            ].precision
                        ),
                        "r": str(
                            get_rouge_scores(reference["sent"], candidate)[
                                "rougeL"
                            ].recall
                        ),
                    },
                }
            elif metric == "BLEU":
                item["metric"] = {
                    "bleu": {
                        "score": get_bleu_score(
                            reference["sent"], candidate, chencherry
                        )
                    }
                }
            elif metric == "BERTScore":
                item["metric"] = {
                    "bertscore": {
                        "f1": get_bert_score(reference["sent"], candidate)["f1"],
                        "p": get_bert_score(reference["sent"], candidate)["precision"],
                        "r": get_bert_score(reference["sent"], candidate)["recall"],
                    }
                }
            print(f'Appending item to data["SocialIQa"]{metric}')
            data["COM2SENSE"][metric].append(item)


# Write the filled JSON structure to the file
with open(result_set_file_path, "w") as file:
    json.dump(data, file, indent=4)
    print("Done!")
