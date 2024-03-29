from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from pathlib import Path
from rouge import Rouge
from bert_score import score
import torch
import matplotlib.ticker as mticker


# Define constants
DATASET_NAMES = ['SocialIQa', 'CoQA', 'COM2SENSE']
DEFAULT_COLORS = ['#4179a3', '#52ce90',
                  '#dc626d', '#c5ced1', '#de9b35', '#e8825d']

FILE_NAMES = ['social_iq_score_set.json',
              'coqa_score_set.json', 'com2sense_score_set.json']
QUALITY_DIMENSIONS = ['veryLow', 'low', 'medium', 'high', 'veryHigh']

BASEDIR = Path(__file__).parents[2]


def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def _calculate_average_rouge_l(data, dataset_name):
    rouge = Rouge()
    values = []
    for dimension in QUALITY_DIMENSIONS:
        hypothesis_sentences = [item['reference_text']
                                for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
        reference_sentences = [item['candidate_text']
                               for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
        scores = rouge.get_scores(
            hypothesis_sentences, reference_sentences, avg=True)
        values.append(scores['rouge-l']['f'])
    print(f'ROUGE-L scores: {values}')
    return values


def _get_bleu_scores(data, dataset_name) -> dict:
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
    print(f'BLEU scores: {bleu_scores}')
    return bleu_scores


def _get_bert_score(data, dataset_name) -> dict:
    values = []
    for dimension in QUALITY_DIMENSIONS:
        hypothesis_sentences = [item['reference_text']
                                for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
        reference_sentences = [item['candidate_text']
                               for item in data[dataset_name]['ROUGE'] if item['model_parameter_group'] == dimension]
        _, _, F1 = score(hypothesis_sentences,
                         reference_sentences, lang='en', verbose=False)
        average_F1 = torch.mean(F1)
        values.append(average_F1)
    print(f'BERT scores: {values}')
    return values


fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[
                       1, 1], height_ratios=[1, 1])

# Create the subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

for id, color in zip([0, 1, 2], DEFAULT_COLORS):
    result_set_file_path = os.path.join(
        BASEDIR, "para_sets/", f"{FILE_NAMES[id]}"
    )

    data = load_data(result_set_file_path)

    dataset_name = DATASET_NAMES[id]

    ax1.plot(QUALITY_DIMENSIONS, _calculate_average_rouge_l(
        data, dataset_name), color=color)

    ax2.plot(QUALITY_DIMENSIONS, list(_get_bleu_scores(
        data, dataset_name).values()), color=color)

    ax3.plot(QUALITY_DIMENSIONS, _get_bert_score(
        data, dataset_name), color=color)

# Set y-axis labels to have two decimal places
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax3.set_ylim(0.77, 0.96)


ax1.set_title('ROUGE-L', fontsize=10.5)
ax1.set_ylabel('F1', fontsize=9)
ax2.set_title('BLEU', fontsize=10.5)
ax3.set_title('BERTscore', fontsize=10.5)
ax3.set_ylabel('F1', fontsize=9)

legend_elements = [Patch(color=c, lw=4, label=l)
                   for c, l in zip(DEFAULT_COLORS, DATASET_NAMES)]

fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize='x-small', edgecolor='lightgrey')

plt.savefig(os.path.join(BASEDIR, 'src/final_scores.png'))

"""
SocialIQa
ROUGE-L scores: [0.514072263516244, 0.43642699136098767, 0.44625214634787813, 0.33657064645534057, 0.45692469184296364]
BLEU scores: {'veryLow': 0.17135089106364895, 'low': 0.09982848586223189, 'medium': 0.11386160992466997, 'high': 0.054695276891230056, 'veryHigh': 0.11928528857866033}
BERT scores: [tensor(0.9280), tensor(0.9165), tensor(0.9176), tensor(0.9018), tensor(0.9193)]

CoQA
ROUGE-L scores: [0.5870273801363547, 0.4287821174154355, 0.3885665119470412, 0.27299950857458655, 0.39705325306311134]
BLEU scores: {'veryLow': 0.22139331171019222, 'low': 0.11881885991816528, 'medium': 0.11133686662191032, 'high': 0.06811153059500391, 'veryHigh': 0.1089131789913937}
BERT scores: [tensor(0.9611), tensor(0.9433), tensor(0.9320), tensor(0.9093), tensor(0.9380)]

COM2SENSE
ROUGE-L scores: [0.5684023066036842, 0.4833325686391094, 0.4493628653720468, 0.3693298906887459, 0.4767609355537734]
BLEU scores: {'veryLow': 0.2152603970315278, 'low': 0.13313002344846364, 'medium': 0.13301166027912834, 'high': 0.07469943510424602, 'veryHigh': 0.1415307294941861}
BERT scores: [tensor(0.9447), tensor(0.9331), tensor(0.9280), tensor(0.9127), tensor(0.9323)]
"""
