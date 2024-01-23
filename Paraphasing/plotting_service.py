import pandas as pd
import matplotlib.pyplot as plt
import json
import os

result_set_file_path = os.path.join(os.path.dirname(__file__), 'social_iq_score_set.json')


with open(result_set_file_path) as f:
    data = json.load(f)

id = data['SocialIQa']['ROUGE'][0]['id']

def get_rougeL_scores() -> dict:
  rougeL_scores = {
    "low": "",
    "medium": "",
    "high": ""
  }
  for item in data['SocialIQa']['ROUGE']:
    if item['id'] == id and item['model_parameter_group'] == "low":
      rougeL_scores['low'] = float(item['metric']['rougeL']['f1'])
    elif item['id'] == id and item['model_parameter_group'] == "medium":
      rougeL_scores['medium'] = float(item['metric']['rougeL']['f1'])
    elif item['id'] == id and item['model_parameter_group'] == "high":
      rougeL_scores['high'] = float(item['metric']['rougeL']['f1'])
  return rougeL_scores

def get_bleu_scores() -> dict:
  bleu_scores = {
    "low": "",
    "medium": "",
    "high": ""
  }
  for item in data['SocialIQa']['BLEU']:
    if item['id'] == id and item['model_parameter_group'] == "low":
      bleu_scores['low'] = float(item['metric']['bleu']['score'])
    elif item['id'] == id and item['model_parameter_group'] == "medium":
      bleu_scores['medium'] = float(item['metric']['bleu']['score'])
    elif item['id'] == id and item['model_parameter_group'] == "high":
      bleu_scores['high'] = float(item['metric']['bleu']['score'])
  return bleu_scores

def get_bert_score() -> dict:
  bert_scores = {
    "low": "",
    "medium": "",
    "high": ""
  }
  for item in data['SocialIQa']['BERTScore']:
    if item['id'] == id and item['model_parameter_group'] == "low":
      bert_scores['low'] = float(item['metric']['bertscore']['f1'])
    elif item['id'] == id and item['model_parameter_group'] == "medium":
      bert_scores['medium'] = float(item['metric']['bertscore']['f1'])
    elif item['id'] == id and item['model_parameter_group'] == "high":
      bert_scores['high'] = float(item['metric']['bertscore']['f1'])
  return bert_scores


# Plot the scores
plt.figure(figsize=(10, 6))
plt.subplot(131)
plt.bar(['low', 'medium', 'high'], list(get_rougeL_scores().values()))
plt.title('ROUGE-L F1')
plt.subplot(132)
plt.bar(['low', 'medium', 'high'], list(get_bleu_scores().values()))
plt.title('BLEU')
plt.subplot(133)
plt.bar(['low', 'medium', 'high'], list(get_bert_score().values()))
plt.title('BERTScore F1')
plt.suptitle('Scores of ROUGE, BLEU and BERTScore for item ' + id)
#plt.grid(True)
plt.show()