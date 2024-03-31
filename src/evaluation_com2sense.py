import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from pathlib import Path
from tasks.com2sense_task import Com2SenseTask
import bigbench.models.huggingface_models as huggingface_models


BASEDIR = Path(__file__).parents[1]


def _score_data_to_dict(score_data):
    return {
        f'score_dict {score_data.number_of_shots}': score_data.score_dict,
        'preferred_score': score_data.preferred_score,
        'number_of_shots': score_data.number_of_shots,
        'low_score': score_data.low_score,
        'high_score': score_data.high_score,
        'subtask_description': score_data.subtask_description
    }


def _load_data(path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _write_data(data, path) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# Make sure to set the correct directory to raw_data variable in the Com2SenseTask class
current_task = Com2SenseTask()


def evaluate_original():
    result_file_path = os.path.join(
        BASEDIR, 'results_no_paraphrase/result_scores.json')

    result_data = _load_data(result_file_path)

    for model in ['gpt2-large', 'gpt2-xl', 'openai-gpt']:
        model = huggingface_models.BIGBenchHFModel(model)
        print(f'Running model {model}')

        score_data = current_task.evaluate_model(model)
        print(f'Final score: {score_data}')

        result_data['COM2SENSE'][model].append(
            _score_data_to_dict(score_data))

    # Write the filled JSON structure to the file
    _write_data(result_data, result_file_path)


def evaluate_paraphrase():
    result_file_path = os.path.join(
        BASEDIR, 'results_paraphrase/result_para_scores.json')

    result_data = _load_data(result_file_path)

    for model in ['gpt2-large', 'gpt2-xl', 'openai-gpt']:
        model = huggingface_models.BIGBenchHFModel(model)
        print(f'Running model {model}')

        score_data = current_task.evaluate_model(model)
        print(f'Final score: {score_data}')

        result_data['COM2SENSE'][model].append(
            _score_data_to_dict(score_data))

    # Write the filled JSON structure to the file
    _write_data(result_data, result_file_path)


if __name__ == "__main__":
    evaluate_original()
    # TODO: Set the correct directory to raw_data variable in the Com2SenseTask class 
    # evaluate_paraphrase()
