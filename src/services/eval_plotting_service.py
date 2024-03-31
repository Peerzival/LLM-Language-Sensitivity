import os
import json
from pathlib import Path
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np


# Define constants
DATASET_NAMES = ['SocialIQa', 'COM2SENSE', 'CoQA']
MODEL_NAMES = ['openai-gpt', 'gpt2-large', 'gpt2-xl']
MODEL_ALIAS = ['GPT-1', 'GPT-2L', 'GPT-2XL',
               'GPT-1+Para', 'GPT-2L+Para', 'GPT-2XL+Para']
SHOTS = [0, 1, 2, 3]
DEFAULT_COLORS = ['#5F72BC', '#91D679',
                  '#FF3383', '#33DAFF', '#EFED19', '#FFA833']


# Define paths
dataset_path = os.path.join(
    Path(__file__).parents[2], 'results_no_paraphrase/result_scores.json')

para_dataset_path = os.path.join(
    Path(__file__).parents[2], 'results_paraphrase/result_para_scores.json')


def _load_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_scores(data, dataset_name, model_name, metric):
    return [item[f'score_dict {i}'][metric] for i, item in enumerate(data[dataset_name][model_name])]


def plot_scores(x, y, color, label, offset):
    plt.bar([i + offset for i in range(len(x))],
            y, color=color, label=label, width=0.1)


def format_extracted_data(dataset, MODEL_NAMES, extract_scores, data, para_data, metric, convert_to_percentage=False):
    y_data = []
    y_para_data = []
    for model_name in MODEL_NAMES:
        y_data.append(extract_scores(
            data, dataset, model_name, metric))
        y_para_data.append(extract_scores(
            para_data, dataset, model_name, metric))
        
    if convert_to_percentage:
        for y_d, y_p in zip(y_data, y_para_data):
            if len(y_d) == 1:
                y_d[0] = y_d[0] * 100
                y_p[0] = y_p[0] * 100
            else:
                for i in range(len(y_d)):
                    y_d[i] = y_d[i] * 100
                    y_p[i] = y_p[i] * 100

    return {
        'GPT-1': y_data[0],
        'GPT-2L': y_data[1],
        'GPT-2XL': y_data[2],
        'GPT-1(P)': y_para_data[0],
        'GPT-2L(P)': y_para_data[1],
        'GPT-2XL(P)': y_para_data[2]
    }


# Load data
data = _load_data(dataset_path)
para_data = _load_data(para_dataset_path)


x = np.arange(len(SHOTS)) * 2
width = 0.31  # the width of the bars
multiplier = 0  # Multiplier to adjust the position of the bars

fig = plt.figure(figsize=(8, 6))
# Create a grid of subplots with custom widths and heights
gs = GridSpec(1, 1, figure=fig)

# Create the subplots
ax1 = fig.add_subplot(gs[0, 0])


def _plot_few_shot_data(dataset_name, MODEL_NAMES, SHOTS, DEFAULT_COLORS, extract_scores, data, para_data, x, width, multiplier, axs, metric, label, y_lim: tuple[int, int] = None):
    data_dict = format_extracted_data(
        dataset_name, MODEL_NAMES, extract_scores, data, para_data, metric, convert_to_percentage=True)
    
    n_bars = len(data_dict)  # Number of bars per group
    total_width = n_bars * width  # Total width of a group of bars

    for attribute, color in zip(data_dict.items(), DEFAULT_COLORS):
        offset = width * multiplier
        bars = axs.bar(x + offset, attribute[1],
                width, color=color, label=attribute[0])
        # axs.bar_label(bars, padding=3, labels=[
        #     f'{v:.1f}' for v in attribute[1]], fontsize=10)
        multiplier += 1

    axs.set_title('(a) Social IQa.', fontsize=20)
    axs.set_ylabel('Multiple Choice Grade', fontsize=18)
    axs.set_xticks(x + total_width / 2 - width / 2, SHOTS)  # Adjust the xticks
    # plt.xticks(rotation=45, ha='right')
    axs.xaxis.set_tick_params(pad=1)
    axs.tick_params(axis='x', labelsize=18)
    axs.tick_params(axis='y', labelsize=18)
    axs.set_xlabel(label, fontsize=18)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

     # Create a list to hold the legend elements
    legend_elements = [Patch(color=c, lw=2, label=l) for c, l in zip(DEFAULT_COLORS, MODEL_ALIAS)]
    # Position the legend at the top center of the figure
    axs.legend(handles=legend_elements, loc='lower right', fontsize='xx-large', edgecolor='lightgrey')

    if y_lim:
        axs.set_ylim(y_lim[0], y_lim[1])


def _plot_zero_data_data(dataset_name, MODEL_NAMES, DEFAULT_COLORS, extract_scores, data, para_data, width, multiplier, axs, metric, label, y_lim: tuple[int, int] = None, show_legend=True):

    if dataset_name == 'COM2SENSE':
        data_dict = format_extracted_data(
        dataset_name, MODEL_NAMES, extract_scores, data, para_data, metric, convert_to_percentage=True)
    else:
        data_dict = format_extracted_data(
        dataset_name, MODEL_NAMES, extract_scores, data, para_data, metric)

    n_bars = len(data_dict)  # Number of bars per group
    total_width = n_bars * width  # Total width of a group of bars
    
    for attribute, color in zip(data_dict.items(), DEFAULT_COLORS):
        offset = width * multiplier
        bars = axs.bar(offset, attribute[1][0],
                width, color=color, label=attribute[0])
        # axs.bar_label(bars, padding=3, labels=[
        #     f'{v:.2f}' for v in attribute[1]], fontsize=6)
        multiplier += 1

    if dataset_name == 'COM2SENSE' and metric == 'standard-accuracy':
        axs.set_ylabel('Standard accuracy', fontsize=19)
    elif dataset_name == 'COM2SENSE' and metric == 'pair-wise-accuracy':
        axs.set_ylabel('Pairwise accuracy', fontsize=19)
    else:
        axs.set_ylabel('F1', fontsize=19)
    axs.set_xticks([total_width / 2 - width / 2])  # Set the xtick in the middle of the x-axis
    axs.set_xticklabels(['Zero-shot'])
    axs.xaxis.set_tick_params(pad=1)
    axs.tick_params(axis='x', labelsize=19)  # Set the label for the xtick
    axs.tick_params(axis='y', labelsize=19)  # Set the label for the xtick
    axs.set_title(label, fontsize=20)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if show_legend:
        # Create a list to hold the legend elements
        legend_elements = [Patch(color=c, lw=4, label=l) for c, l in zip(DEFAULT_COLORS, MODEL_ALIAS)]
        # Position the legend at the top center of the figure
        axs.legend(handles=legend_elements, loc='upper right', fontsize='xx-large', edgecolor='lightgrey')

    if y_lim:
        axs.set_ylim(y_lim[0], y_lim[1])


_plot_zero_data_data(DATASET_NAMES[1], MODEL_NAMES, DEFAULT_COLORS,
                     extract_scores, data, para_data, width, multiplier, ax1, metric='standard-accuracy', label= '(c) COM2SENSE.', y_lim=(0, 100), show_legend=False)

_plot_zero_data_data(DATASET_NAMES[1], MODEL_NAMES, DEFAULT_COLORS,
                     extract_scores, data, para_data, width, multiplier, ax2, metric='pair-wise-accuracy', label= '(c) COM2SENSE.', y_lim=(0, 100))

_plot_zero_data_data(DATASET_NAMES[2], MODEL_NAMES, DEFAULT_COLORS,extract_scores, data, para_data, width, multiplier, ax1, metric='f1', label='(b) CoQA.', y_lim=(0, 20))

_plot_few_shot_data(DATASET_NAMES[0], MODEL_NAMES, SHOTS, DEFAULT_COLORS,
                    extract_scores, data, para_data, x, width, multiplier, ax1, metric='multiple_choice_grade', label='Number of shots.', y_lim=(0, 50))

# fig.suptitle('(c) COM2SENSE.', fontsize=20)
fig.tight_layout()

plt.show()
# plt.savefig('coqa_results.pdf', format='pdf', dpi=600, bbox_inches='tight')
