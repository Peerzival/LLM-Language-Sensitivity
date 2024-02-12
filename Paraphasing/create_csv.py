import json
import csv
import os


score_to_metric = {
    'ROUGE': ['rouge1', 'rouge2', 'rougeL'],
    'BLEU': ['bleu'],
    'BERTScore': ['bertscore']
}

dataset_id = [0, 1, 2]
json_file_names = ['social_iq_score_set.json',
                   'coqa_score_set.json', 'com2sense_score_set.json']
csv_file_names = ['social_iqa.csv', 'coqa.csv', 'com2sense.csv']
dataset_names = ['SocialIQa', 'CoQA', 'COM2SENSE']


def get_items_with_id(data, item_id):
    rouge_items = [item for item in data['ROUGE'] if item['id'] == item_id]
    bleu_items = [item for item in data['BLEU'] if item['id'] == item_id]
    bertscore_items = [item for item in data['BERTScore']
                       if item['id'] == item_id]

    for item in single_item(rouge_items, bleu_items, bertscore_items):
        yield item


def single_item(rouge_items, bleu_items, bertscore_items):
    bleu_dict = {(item['id'], item['model_parameter_group'])                 : item for item in bleu_items}
    bert_dict = {(item['id'], item['model_parameter_group'])                 : item for item in bertscore_items}

    for rouge_item in rouge_items:
        bleu_item = bleu_dict.get(
            (rouge_item['id'], rouge_item['model_parameter_group']), None)
        bert_item = bert_dict.get(
            (rouge_item['id'], rouge_item['model_parameter_group']), None)
        yield {
            "reference_text": rouge_item['reference_text'],
            "candidate_text": rouge_item['candidate_text'],
            "model_parameter_group": rouge_item['model_parameter_group'],
            "ROUGE": rouge_item['metric'],
            "BLEU": bleu_item['metric'] if bleu_item else None,
            "BERTscore": bert_item['metric'] if bert_item else None
        }


def get_item_stats(data, dataset_name):
    finished_item = any
    for item in data[dataset_name]['ROUGE']:
        if finished_item != item['id']:
            for stat_item in get_items_with_id(data[dataset_name], item['id']):
                rouge = stat_item['ROUGE']
                bleu = stat_item['BLEU']
                bertscore = stat_item['BERTscore']
                yield [
                    remove_special_chars(stat_item['reference_text']),
                    stat_item['candidate_text'],
                    stat_item['model_parameter_group'],
                    rouge['rouge1']['f1'],
                    rouge['rouge1']['p'],
                    rouge['rouge1']['r'],
                    rouge['rouge2']['f1'],
                    rouge['rouge2']['p'],
                    rouge['rouge2']['r'],
                    rouge['rougeL']['f1'],
                    rouge['rougeL']['p'],
                    rouge['rougeL']['r'],
                    bleu['bleu']['score'] if bleu else None,
                    bertscore['bertscore']['f1'] if bertscore else None,
                    bertscore['bertscore']['p'] if bertscore else None,
                    bertscore['bertscore']['r'] if bertscore else None
                ]
            finished_item = item['id']


def remove_special_chars(text):
    # Define special characters
    special_chars = "\u201c\u201d\u2019\u2026\n\n\n"

    # Create a translation table that maps every special character to None
    trans_table = str.maketrans("", "", special_chars)

    # Use the translation table to remove special characters from the text
    clean_text = text.translate(trans_table)

    return clean_text


if __name__ == "__main__":
    for id in dataset_id:
        dataset_name = dataset_names[id]

        result_set_file_path = os.path.join(
            os.path.dirname(__file__), "para_sets/", f"{json_file_names[id]}"
        )

        csvc_file_path = os.path.join(
            os.path.dirname(
                __file__), "para_sets_csv/", f"{csv_file_names[id]}"
        )

        with open(result_set_file_path) as json_file:
            data = json.load(json_file)

        with open(csvc_file_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for item_stat in get_item_stats(data, dataset_name):
                writer.writerow(item_stat)
