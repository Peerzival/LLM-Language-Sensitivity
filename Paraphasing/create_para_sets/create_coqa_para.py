import json
import os
import json

from pathlib import Path
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from bert_score import score
from paraphrasing_model import QualityControlPipeline

BASEDIR = Path(__file__).parents[1]
data_set_file_path = os.path.join(
    BASEDIR, "Dataset/CoQA/Full_set/coqa.test.json")
result_set_file_path = os.path.join(
    BASEDIR, "para_sets/coqa_score_set.json"
)
parameter_groups_file_path = os.path.join(
    os.path.dirname(__file__), "parameter_groups.json"
)


def _get_rouge_scores(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)


def _get_bleu_score(
    reference: str, candidate: str, smoothingFunction: SmoothingFunction
) -> str:
    reference = reference.split()
    candidate = candidate.split()
    return sentence_bleu(
        [reference], candidate, smoothing_function=smoothingFunction.method1
    )


def _get_bert_score(reference: str, candidate: str) -> dict:
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
        for question in reference["questions"]:
            candidate = model(
                question["input_text"],
                parameter_set[parameter]["lexical"],
                parameter_set[parameter]["syntactic"],
                parameter_set[parameter]["semantic"],
            )
            candidate = candidate[0]["generated_text"]
            # Fill the JSON structure
            for metric in ["ROUGE", "BLEU", "BERTScore"]:
                item = {
                    "id": reference["id"],
                    "reference_text": question["input_text"],
                    "candidate_text": candidate,
                    "model_parameter_group": parameter,
                    "metric": {},
                }
                if metric == "ROUGE":
                    rouge_scores = _get_rouge_scores(
                        question["input_text"], candidate)
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
                            "score": _get_bleu_score(
                                question["input_text"], candidate, chencherry
                            )
                        }
                    }
                elif metric == "BERTScore":
                    bert_scores = _get_bert_score(
                        question["input_text"], candidate)
                    item["metric"] = {
                        "bertscore": {
                            "f1": bert_scores["f1"],
                            "p": bert_scores["precision"],
                            "r": bert_scores["recall"],
                        }
                    }
                data["CoQA"][metric].append(item)

print("Finished processing the CoQA dataset")

# Write the filled JSON structure to the file
with open(result_set_file_path, "w") as file:
    json.dump(data, file, indent=4)
