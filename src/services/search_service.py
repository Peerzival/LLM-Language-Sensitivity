import json
import os
import numpy as np
from pathlib import Path
from rouge import Rouge
from bert_score import score
import torch
from multiprocessing import Pool
import multiprocessing

def load_data(path):
    with open(path, 'r') as f:
        result_data = json.load(f)
    return result_data

def calculate_metrics(args):
    data, item_id, dataset_name = args
    rouge = Rouge()
    hypothesis_sentences = []
    reference_sentences = []
    medium = []
    rouge_data = data[dataset_name]['ROUGE']
    bleu_data = data[dataset_name]['BLEU']
    for item in rouge_data:
        if item['model_parameter_group'] == 'medium' and item['id'] == item_id:
            hypothesis_sentences.append(item['reference_text'])
            reference_sentences.append(item['candidate_text'])
    for item in bleu_data:
        if item['model_parameter_group'] == "medium" and item['id'] == item_id:
            medium.append(item['metric']['bleu']['score'])
    scores = rouge.get_scores(hypothesis_sentences, reference_sentences, avg=True)
    rouge_l = (item_id, scores['rouge-l']['f'])
    bleu = (item_id, np.mean(medium))
    _, _, F1 = score(hypothesis_sentences, reference_sentences, lang='en', verbose=False)
    average_F1 = torch.mean(F1)
    bert = (item_id, average_F1)
    return rouge_l, bleu, bert

BASEDIR = Path(__file__).parents[2]

data_sets = ['SocialIQa', 'CoQA', 'COM2SENSE']

para_data_file_path = os.path.join(
    BASEDIR, 'Paraphasing\para_sets\coqa_score_set.json')

data_file_path = os.path.join(
    BASEDIR, 'Dataset\CoQA\Full_set\coqa.test.json')

if __name__ == "__main__":
    data = load_data(data_file_path)
    para_data = load_data(para_data_file_path)
    lowest_rouge_l_item = (1, 1)
    lowest_bleu_item = (1, 1)
    highest_bert_score_item = (1, 0)

    chunk_size = 6
    chunks = [data['data'][i:i + chunk_size] for i in range(0, len(data['data']), chunk_size)]
    
    for chunk in chunks:
        with Pool() as p:
            results = p.map(calculate_metrics, [(para_data, item['id'], 'CoQA') for item in chunk])
        for rouge_l_item, bleu_item, bert_score_item in results:
            if rouge_l_item[1] < lowest_rouge_l_item[1]:
                lowest_rouge_l_item = rouge_l_item
            if bleu_item[1] < lowest_bleu_item[1]:
                lowest_bleu_item = bleu_item
            if bert_score_item[1] > highest_bert_score_item[1]:
                highest_bert_score_item = bert_score_item
    print(f'Lowest ROUGE-L score: {lowest_rouge_l_item}')
    print(f'Lowest BLEU score: {lowest_bleu_item}')
    print(f'Highest BERTscore: {highest_bert_score_item}')

"""
Lowest ROUGE-L score: ('3vnxk88kkcivuhrv1d113uw1iwwv9w', 0.11149683451988009)
Lowest BLEU score: ('3vnxk88kkcivuhrv1d113uw1iwwv9w', 0.01931965027942906)
Highest BERTscore: ('3dip6yhapcsee1mz1v6d3ud4xjre8b', tensor(0.9587))
"""