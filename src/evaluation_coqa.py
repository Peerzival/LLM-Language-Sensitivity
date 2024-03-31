import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
from tasks.coqa_task import CoQA
import bigbench.models.huggingface_models as huggingface_models
from pathlib import Path
import json


BASEDIR = Path(__file__).parents[1]


def _score_data_to_dict(score_data, number_of_examples):
    return {
        f'score_dict {number_of_examples}': score_data.score_dict,
        'number_of_shots': number_of_examples,
    }


def _load_data(path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _write_data(data, path) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# Make sure to set the correct directory to coqa_test_file_path variable in the CoqaTask class
coqa = CoQA(batch_size=64, max_examples=110, verbose=True)


def evaluate_original():
    result_file_path = os.path.join(BASEDIR, 'results_no_paraphrase/result_scores.json')
    result_data = _load_data(result_file_path)

    for model in ['gpt2-large', 'gpt2-xl', 'openai-gpt']:
        model = huggingface_models.BIGBenchHFModel(model)

        score_data = coqa.evaluate_model(model)
        print(f'Final score for {model}: {score_data}')

        for score in score_data:
            if score.subtask_description == 'CoQA\'s overall score':
                result_data['CoQA'][model].append(
                    _score_data_to_dict(score, -1))

    _write_data(result_data, result_file_path)


def evaluate_paraphrase():
    result_file_path = os.path.join(BASEDIR, 'results_paraphrase/result_para_scores.json')
    result_data = _load_data(result_file_path)

    for model in ['gpt2-large', 'gpt2-xl', 'openai-gpt']:
        model = huggingface_models.BIGBenchHFModel(model)

        score_data = coqa.evaluate_model(model)
        print(f'Final score for {model}: {score_data}')

        for score in score_data:
            if score.subtask_description == 'CoQA\'s overall score':
                result_data['CoQA'][model].append(
                    _score_data_to_dict(score, -1))

    _write_data(result_data, result_file_path)


if __name__ == "__main__":
    evaluate_original()
    # TODO: Set the correct directory to coqa_test_file_path variable in the CoqaTask class
    # evaluate_paraphrase()
