import json
from pathlib import Path
from bigbench.api import json_task
import bigbench.models.huggingface_models as huggingface_models
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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


def _json_task(data):
    return json_task.JsonTask(
        task_data=data,
        shot_list=[0, 1, 2, 3],
        verbose=True
    )


def evaluate_original():
    print('Evaluating original data')
    data_set_file_path = os.path.join(
        BASEDIR, 'datasets/Social_IQa/Original/socialIqa.json')

    result_file_path = os.path.join(
        BASEDIR, 'results_no_paraphrase/result_scores.json')

    data = _load_data(data_set_file_path)
    current_task = _json_task(data)
    result_data = _load_data(result_file_path)

    for model in ['gpt2-large', 'gpt2-xl', 'openai-gpt']:
        model = huggingface_models.BIGBenchHFModel(model)

        score_data = current_task.evaluate_model(model)
        print(f'Final score: {score_data}')

        for score in score_data:
            result_data['SocialIQa'][model].append(
                _score_data_to_dict(score))

    # Write the filled JSON structure to the file
    _write_data(result_data, result_file_path)
    print('Finished evaluating original data')


def evaluate_paraphrase():
    print('Evaluating paraphrased data')
    data_set_file_path = os.path.join(
        BASEDIR, 'datasets/Social_IQa/Evaluation/socialIqa_para.json')

    result_file_path = os.path.join(
        BASEDIR, 'results_paraphrase/result_para_scores.json')

    data = _load_data(data_set_file_path)
    current_task = _json_task(data)
    result_data = _load_data(result_file_path)

    for model in ['gpt2-large', 'gpt2-xl', 'openai-gpt']:
        model = huggingface_models.BIGBenchHFModel(model)

        score_data = current_task.evaluate_model(model)
        print(f'Final score: {score_data}')

        for score in score_data:
            result_data['SocialIQa'][model].append(
                _score_data_to_dict(score))

    # Write the filled JSON structure to the file
    _write_data(result_data, result_file_path)
    print('Finished evaluating paraphrased data')


if __name__ == "__main__":
    evaluate_original()
    evaluate_paraphrase()
