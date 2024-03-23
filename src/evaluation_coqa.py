import json
from pathlib import Path
import bigbench.models.huggingface_models as huggingface_models
import os

from tasks.coqa_task import CoQA


BASEDIR = Path(__file__).parents[1]


def score_data_to_dict(score_data, number_of_examples):
    return {
        f'score_dict {number_of_examples}': score_data.score_dict,
        # 'preferred_score': score_data.preferred_score,
        'number_of_shots': number_of_examples,
        # 'low_score': score_data.low_score,
        # 'high_score': score_data.high_score,
        # 'subtask_description': score_data.subtask_description
    }


result_file_path = os.path.join(
    BASEDIR, 'results_no_paraphrase/result_scores.json')

data_sets = ['SocialIQa', 'CoQA', 'COM2SENSE']


if __name__ == "__main__":

    with open(result_file_path, 'r') as f:
        result_data = json.load(f)

    model_types = ['gpt2-large', 'gpt2-xl', 'openai-gpt']

    coqa = CoQA(batch_size=10, verbose=False)

    model = huggingface_models.BIGBenchHFModel(model_types[0])

    score_data = coqa.evaluate_model(model)
    print(f'Final score for {model_types[0]}: {score_data}')

    for score in score_data:
        if score.subtask_description == 'CoQA\'s overall score':
            result_data[data_sets[1]][model_types[0]].append(
                score_data_to_dict(score, -1))

    with open(result_file_path, 'w') as f:
        json.dump(result_data, f, indent=4)
