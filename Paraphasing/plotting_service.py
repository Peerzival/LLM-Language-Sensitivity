import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from pathlib import Path

dataset_id = [0, 1, 2]
file_names = ['social_iq_score_set.json',
              'coqa_score_set.json', 'com2sense_score_set.json']
dataset_names = ['SocialIQa', 'CoQA', 'COM2SENSE']

for id in dataset_id:
    result_set_file_path = os.path.join(
        os.path.dirname(__file__), "para_sets/", f"{file_names[id]}"
    )

    with open(result_set_file_path) as f:
        data = json.load(f)
    # id = data[dataset_name]['ROUGE'][0]['id']

    dataset_name = dataset_names[id]

    def get_rougeL_scores() -> dict:
        rougeL_scores = {
            "veryLow": "",
            "low": "",
            "medium": "",
            "high": "",
            "veryHigh": ""
        }

        veryLow: float = 0.0
        low: float = 0.0
        medium: float = 0.0
        high: float = 0.0
        veryHigh: float = 0.0

        # for item in data[dataset_name]['ROUGE']:
        #   if item['id'] == id and item['model_parameter_group'] == "low":
        #     rougeL_scores['low'] = float(item['metric']['rougeL']['f1'])
        #   elif item['id'] == id and item['model_parameter_group'] == "medium":
        #     rougeL_scores['medium'] = float(item['metric']['rougeL']['f1'])
        #   elif item['id'] == id and item['model_parameter_group'] == "high":
        #     rougeL_scores['high'] = float(item['metric']['rougeL']['f1'])

        for item in data[dataset_name]['ROUGE']:
            if item['model_parameter_group'] == "veryLow":
                veryLow += item['metric']['rougeL']['f1']
            elif item['model_parameter_group'] == "low":
                low += item['metric']['rougeL']['f1']
            elif item['model_parameter_group'] == "medium":
                medium += item['metric']['rougeL']['f1']
            elif item['model_parameter_group'] == "high":
                high += item['metric']['rougeL']['f1']
            elif item['model_parameter_group'] == "veryHigh":
                veryHigh += item['metric']['rougeL']['f1']

        rougeL_scores['veryLow'] = veryLow / len(data[dataset_name]['ROUGE'])
        rougeL_scores['low'] = low / len(data[dataset_name]['ROUGE'])
        rougeL_scores['medium'] = medium / len(data[dataset_name]['ROUGE'])
        rougeL_scores['high'] = high / len(data[dataset_name]['ROUGE'])
        rougeL_scores['veryHigh'] = veryHigh / len(data[dataset_name]['ROUGE'])

        return rougeL_scores

    def get_bleu_scores() -> dict:
        bleu_scores = {
            "veryLow": "",
            "low": "",
            "medium": "",
            "high": "",
            "veryHigh": ""
        }

        veryLow: float = 0.0
        low: float = 0.0
        medium: float = 0.0
        high: float = 0.0
        veryHigh: float = 0.0

        # for item in data[dataset_name]['BLEU']:
        #   if item['id'] == id and item['model_parameter_group'] == "low":
        #     bleu_scores['low'] = float(item['metric']['bleu']['score'])
        #   elif item['id'] == id and item['model_parameter_group'] == "medium":
        #     bleu_scores['medium'] = float(item['metric']['bleu']['score'])
        #   elif item['id'] == id and item['model_parameter_group'] == "high":
        #     bleu_scores['high'] = float(item['metric']['bleu']['score'])

        for item in data[dataset_name]['BLEU']:
            if item['model_parameter_group'] == "veryLow":
                veryLow += item['metric']['bleu']['score']
            elif item['model_parameter_group'] == "low":
                low += item['metric']['bleu']['score']
            elif item['model_parameter_group'] == "medium":
                medium += item['metric']['bleu']['score']
            elif item['model_parameter_group'] == "high":
                high += item['metric']['bleu']['score']
            elif item['model_parameter_group'] == "veryHigh":
                veryHigh += item['metric']['bleu']['score']

        bleu_scores['veryLow'] = veryLow / len(data[dataset_name]['BLEU'])
        bleu_scores['low'] = low / len(data[dataset_name]['BLEU'])
        bleu_scores['medium'] = medium / len(data[dataset_name]['BLEU'])
        bleu_scores['high'] = high / len(data[dataset_name]['BLEU'])
        bleu_scores['veryHigh'] = veryHigh / len(data[dataset_name]['BLEU'])

        return bleu_scores

    def get_bert_score() -> dict:
        bert_scores = {
            "veryLow": "",
            "low": "",
            "medium": "",
            "high": "",
            "veryHigh": ""
        }

        veryLow: float = 0.0
        low: float = 0.0
        medium: float = 0.0
        high: float = 0.0
        veryHigh: float = 0.0

        # for item in data[dataset_name]['BERTScore']:
        #   if item['id'] == id and item['model_parameter_group'] == "low":
        #     bert_scores['low'] = float(item['metric']['bertscore']['f1'])
        #   elif item['id'] == id and item['model_parameter_group'] == "medium":
        #     bert_scores['medium'] = float(item['metric']['bertscore']['f1'])
        #   elif item['id'] == id and item['model_parameter_group'] == "high":
        #     bert_scores['high'] = float(item['metric']['bertscore']['f1'])

        for item in data[dataset_name]['BERTScore']:
            if item['model_parameter_group'] == "veryLow":
                veryLow += float(item['metric']['bertscore']['f1'])
            elif item['model_parameter_group'] == "low":
                low += float(item['metric']['bertscore']['f1'])
            elif item['model_parameter_group'] == "medium":
                medium += float(item['metric']['bertscore']['f1'])
            elif item['model_parameter_group'] == "high":
                high += float(item['metric']['bertscore']['f1'])
            elif item['model_parameter_group'] == "veryHigh":
                veryHigh += float(item['metric']['bertscore']['f1'])

        bert_scores['veryLow'] = veryLow / len(data[dataset_name]['BERTScore'])
        bert_scores['low'] = low / len(data[dataset_name]['BERTScore'])
        bert_scores['medium'] = medium / len(data[dataset_name]['BERTScore'])
        bert_scores['high'] = high / len(data[dataset_name]['BERTScore'])
        bert_scores['veryHigh'] = veryHigh / \
            len(data[dataset_name]['BERTScore'])

        return bert_scores

    x = ['very low', 'low', 'medium', 'high', 'very high']
    # x = ['low', 'medium', 'high']

    # Plot the scores
    plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(x, list(get_rougeL_scores().values()))
    ax0.set_title('Avrg. ROUGE-L F1')
    # ax0.set_xlabel('Model Parameter Group')

    ax1 = plt.subplot(gs[1, 0])
    ax1.plot(x, list(get_bleu_scores().values()))
    ax1.set_title('Avrg. BLEU')

    ax2 = plt.subplot(gs[:, 1])
    ax2.plot(x, list(get_bert_score().values()))
    ax2.set_title('Avrg. BERTScore F1')

    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.10,
                        right=0.95, hspace=0.5, wspace=0.35)
    plt.suptitle(
        f'Avrg. scores of ROUGE, BLEU and BERTScore for {dataset_name} dataset', fontsize=14, weight='bold')

    plt.show()
