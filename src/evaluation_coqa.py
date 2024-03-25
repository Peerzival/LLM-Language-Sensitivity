import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from pathlib import Path
import bigbench.models.huggingface_models as huggingface_models
from tasks.coqa_task import CoQA
import time

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
    start_time = time.time()
    with open(result_file_path, 'r') as f:
        result_data = json.load(f)

    model_types = ['gpt2-large', 'gpt2-xl', 'openai-gpt']

    coqa = CoQA(batch_size=64, max_examples=110, verbose=True)

    model = huggingface_models.BIGBenchHFModel(model_types[0])

    print(f'Running {model_types[0]}')
    score_data = coqa.evaluate_model(model)
    print(f'Final score for {model_types[0]}: {score_data}')

    for score in score_data:
        if score.subtask_description == 'CoQA\'s overall score':
            result_data[data_sets[1]][model_types[0]].append(
                score_data_to_dict(score, -1))

    with open(result_file_path, 'w') as f:
        json.dump(result_data, f, indent=4)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The script took {execution_time} seconds to run.")

    """
    Final score for gpt2-large: [ScoreData(score_dict={'exact match': 0.0, 'f1': 0.0}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's children_stories score"), ScoreData(score_dict={'exact match': 16.7, 'f1': 18.3}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's literature score"), ScoreData(score_dict={'exact match': 0.0, 'f1': 9.3}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's mid-high_school score"), ScoreData(score_dict={'exact match': 17.5, 'f1': 22.5}, preferred_score='f1', number_of_shots=0, low0.0, high_score=100.0, subtask_score=0.0, high_score=100.0, subtask_description="CoQA's news score"), ScoreData(score_dict={'exact match': 2.4, 'f1': 3.9}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's reddit score"), ScoreData(score_dict={'exact match': 0.0, 'f1': 1.5}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's science score"), ScoreData(score_dict={'exact match': 6.5, 'f1': 10.5}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's in_domain score"), ScoreData(score_dict={'exact match': 1.9, 'f1': 3.4}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's out_domain score"), ScoreData(score_dict={'exact match': 4.3, 'f1': 7.2}, preferred_score='f1', number_of_shots=0, low_score=0.0, high_score=100.0, subtask_description="CoQA's overall score")]
                {
                "score_dict -1": {
                    "exact match": 4.3,
                    "f1": 7.2
                },
                "number_of_shots": -1
            }
    """