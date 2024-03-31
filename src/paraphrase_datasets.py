import json
import os
import json
import uuid

from pathlib import Path
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from bert_score import score
from models.qcpg_model import QualityControlPipeline


def _load_data(path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _save_data(data, path) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


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


BASEDIR = Path(__file__).parents[1]

parameter_groups_file_path = os.path.join(
    os.path.dirname(__file__), "data/qdgs.json"
)

# Initialize the paraphrasing model
model = QualityControlPipeline("sentences")
# Set the smoothing function for BLEU
chencherry = SmoothingFunction()

# Create the JSON structure
coqa_data = {"CoQA": {"ROUGE": [], "BLEU": [], "BERTScore": []}}
social_iqa_data = {"SocialIQa": {"ROUGE": [], "BLEU": [], "BERTScore": []}}
com2sense_data = {"COM2SENSE": {"ROUGE": [], "BLEU": [], "BERTScore": []}}

parameter_set = _load_data(parameter_groups_file_path)


def create_coqa_qdgs(data: dict) -> None:
    data_set_file_path = os.path.join(
        BASEDIR, "datasets/CoQA/Original/coqa.json")

    result_set_file_path = os.path.join(
        BASEDIR, "outputs/para_sets/coqa_score_set.json"
    )
    dataset = _load_data(data_set_file_path)

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
    _save_data(data, result_set_file_path)


def create_social_iqa_qdgs(data: dict) -> None:
    data_set_file_path = os.path.join(
        BASEDIR, "datasets/Social_IQa/Original/socialIqa.json"
    )
    result_set_file_path = os.path.join(
        os.path.dirname(__file__), "outputs/para_sets/social_iqa_score_set.json"
    )

    dataset = _load_data(data_set_file_path)

    for reference in dataset["examples"]:
        id = str(uuid.uuid4())
        for parameter in parameter_set:
            candidate = model(
                reference["input"],
                parameter_set[parameter]["lexical"],
                parameter_set[parameter]["syntactic"],
                parameter_set[parameter]["semantic"],
            )
            candidate = candidate[0]["generated_text"]
            # Fill the JSON structure
            for metric in ["ROUGE", "BLEU", "BERTScore"]:
                item = {
                    "id": id,
                    "reference_text": reference["input"],
                    "candidate_text": candidate,
                    "model_parameter_group": parameter,
                    "metric": {},
                }
                if metric == "ROUGE":
                    rouge_scores = _get_rouge_scores(
                        reference["input"], candidate)
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
                                reference["input"], candidate, chencherry
                            )
                        }
                    }
                elif metric == "BERTScore":
                    bert_scores = _get_bert_score(
                        reference["input"], candidate)
                    item["metric"] = {
                        "bertscore": {
                            "f1": bert_scores["f1"],
                            "p": bert_scores["precision"],
                            "r": bert_scores["recall"],
                        }
                    }
                data["SocialIQa"][metric].append(item)

    # Write the filled JSON structure to the file
    _save_data(data, result_set_file_path)


def create_com2sense_qdgs(data: dict) -> None:
    data_set_file_path = os.path.join(
    BASEDIR, "datasets/COM2SENSE/COM2SENSE.json"
    )
    result_set_file_path = os.path.join(
        os.path.dirname(__file__), "outputs/para_sets/com2sense_score_set.json"
    )

    dataset = _load_data(data_set_file_path)

    for reference in dataset["examples"]:
        for parameter in parameter_set:
            candidate = model(
                reference["sent"],
                parameter_set[parameter]["lexical"],
                parameter_set[parameter]["syntactic"],
                parameter_set[parameter]["semantic"],
            )
            candidate = candidate[0]["generated_text"]
            # Fill the JSON structure
            for metric in ["ROUGE", "BLEU", "BERTScore"]:
                item = {
                    "id": reference["id"],
                    "reference_text": reference["sent"],
                    "candidate_text": candidate,
                    "model_parameter_group": parameter,
                    "metric": {},
                }
                if metric == "ROUGE":
                    rouge_scores = _get_rouge_scores(reference["sent"], candidate)
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
                                reference["sent"], candidate, chencherry
                            )
                        }
                    }
                elif metric == "BERTScore":
                    bert_scores = _get_bert_score(reference["sent"], candidate)
                    item["metric"] = {
                        "bertscore": {
                            "f1": bert_scores["f1"],
                            "p": bert_scores["precision"],
                            "r": bert_scores["recall"],
                        }
                    }
                data["COM2SENSE"][metric].append(item)

    # Write the filled JSON structure to the file
    _save_data(data, result_set_file_path)


if __name__ == "__main__":
    create_coqa_qdgs(coqa_data)
    create_social_iqa_qdgs(social_iqa_data)
    create_com2sense_qdgs(com2sense_data)
