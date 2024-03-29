import json
import os
from pathlib import Path

def load_data(path):
    with open(path, 'r') as f:
        result_data = json.load(f)
    return result_data

def find_best_prompt(data, para_data):
    pass

BASEDIR = Path(__file__).parents[1]

data_sets = ['SocialIQa', 'CoQA', 'COM2SENSE']

para_data_file_path = os.path.join(
    BASEDIR, 'Paraphasing\para_sets\coqa_score_set.json')

data_file_path = os.path.join(
    BASEDIR, 'Dataset\CoQA\Full_set\coqa.test.json')


if __name__ == "__main__":
    data = load_data(data_file_path)
    para_data = load_data(para_data_file_path)
    item_data = {
        'id': '',
        'bleu': 0,
        'rouge_l': 0,
        'bert_score': 0
    }
    lowest_bleu: float = 0
    lowest_rouge_l: float = 0
    highest_bert_score: float = 0

    for item in data['data']:
        for metric in para_data[data_sets[1]]:
            for data in metric:
                if metric == ' ROUGE':
                    if data['model_parameter_group'] == 'medium' and data['id'] == item['id']:
                        pass

