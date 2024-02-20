import json
import os

from bigbench.api import json_task
import bigbench.models.huggingface_models as huggingface_models


def score_data_to_dict(score_data):
    return {
        f'score_dict {score_data.number_of_shots}': score_data.score_dict,
        'preferred_score': score_data.preferred_score,
        'number_of_shots': score_data.number_of_shots,
        'low_score': score_data.low_score,
        'high_score': score_data.high_score,
        'subtask_description': score_data.subtask_description
    }


data_set_file_path = os.path.join(
    os.path.dirname(__file__), 'dataset/Common-sense/Social_IQa/Full_set/socialIqa.json')
# data_set_file_path = os.path.join(
#     os.path.dirname(__file__), 'Dataset/Common-sense/Social_IQa/Social_IQa.json')
result_file_path = os.path.join(
    os.path.dirname(__file__), 'results/scores.json')

if __name__ == "__main__":
    # @param  ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'openai-gpt']
    model_type = 'gpt2-medium'
    model = huggingface_models.BIGBenchHFModel(model_type)
    results = {
        model_type: []
    }

    with open(data_set_file_path, 'r') as file:
        data = json.load(file)
        current_task = json_task.JsonTask(
            task_data=data,
            shot_list=[0, 1, 2, 3]
        )
        score_data = current_task.evaluate_model(model)
        print(f'Final score: {score_data}')

        with open(result_file_path, 'w') as f:
            for score in score_data:
                results[model_type].append(score_data_to_dict(score))
            json.dump(results, f, indent=4)
