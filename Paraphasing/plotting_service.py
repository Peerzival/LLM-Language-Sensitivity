import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from pathlib import Path
from rouge import Rouge
from bert_score import score
import torch

dataset_id = [0, 1, 2]
file_names = ['social_iq_score_set.json',
              'coqa_score_set.json', 'com2sense_score_set.json']
dataset_names = ['SocialIQa', 'CoQA', 'COM2SENSE']
quality_dimensions = ['veryLow', 'low', 'medium', 'high', 'veryHigh']

for id in dataset_id:
    result_set_file_path = os.path.join(
        os.path.dirname(__file__), "para_sets/", f"{file_names[id]}"
    )

    with open(result_set_file_path) as f:
        data = json.load(f)
    # id = data[dataset_name]['ROUGE'][0]['id']

    dataset_name = dataset_names[id]

    def calculate_average_rouge_l():
        rouge = Rouge()
        values = []
        for dimension in quality_dimensions:
            hypothesis_sentences = [item['reference_text']
                                    for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
            reference_sentences = [item['candidate_text']
                                   for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
            scores = rouge.get_scores(
                hypothesis_sentences, reference_sentences, avg=True)
            values.append(scores['rouge-l']['f'])
        return values

    def get_bleu_scores() -> dict:
        bleu_scores = {
            "veryLow": "",
            "low": "",
            "medium": "",
            "high": "",
            "veryHigh": ""
        }

        veryLow = []
        low = []
        medium = []
        high = []
        veryHigh = []

        for item in data[dataset_name]['BLEU']:
            if item['model_parameter_group'] == "veryLow":
                veryLow.append(item['metric']['bleu']['score'])
            elif item['model_parameter_group'] == "low":
                low.append(item['metric']['bleu']['score'])
            elif item['model_parameter_group'] == "medium":
                medium.append(item['metric']['bleu']['score'])
            elif item['model_parameter_group'] == "high":
                high.append(item['metric']['bleu']['score'])
            elif item['model_parameter_group'] == "veryHigh":
                veryHigh.append(item['metric']['bleu']['score'])

        bleu_scores['veryLow'] = sum(veryLow) / len(veryLow)
        bleu_scores['low'] = sum(low) / len(low)
        bleu_scores['medium'] = sum(medium) / len(medium)
        bleu_scores['high'] = sum(high) / len(high)
        bleu_scores['veryHigh'] = sum(veryHigh) / len(veryHigh)

        return bleu_scores

    def get_bert_score() -> dict:
        values = []
        for dimension in quality_dimensions:
            hypothesis_sentences = [item['reference_text']
                                    for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
            reference_sentences = [item['candidate_text']
                                   for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
            P, R, F1 = score(hypothesis_sentences,
                             reference_sentences, lang='en', verbose=False)
            average_F1 = torch.mean(F1)
            values.append(average_F1)

        return values

    # Plot the scores
    plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(quality_dimensions, calculate_average_rouge_l())
    ax0.set_title('Avrg. ROUGE-L F1')
    # ax0.set_xlabel('Model Parameter Group')

    ax1 = plt.subplot(gs[1, 0])
    ax1.plot(quality_dimensions, list(get_bleu_scores().values()))
    ax1.set_title('Avrg. BLEU')

    ax2 = plt.subplot(gs[:, 1])
    ax2.plot(quality_dimensions, get_bert_score())
    ax2.set_title('Avrg. BERTScore F1')

    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.10,
                        right=0.95, hspace=0.5, wspace=0.35)
    plt.suptitle(
        f'Avrg. scores of ROUGE, BLEU and BERTScore for {dataset_name} dataset', fontsize=14, weight='bold')

    plt.show()
