import json
from pathlib import Path

BASEDIR = Path(__file__).parents[1]


def _load_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def _save_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def _filter_items_by_parameter_group(data, dataset: str, parameter_group: str) -> dict:
    return [item["candidate_text"] for item in data[dataset]["ROUGE"] if item["model_parameter_group"] == parameter_group]


def create_coqa_para_set(qdg='medium'):
    print("\nCreating COQA para set")

    data_set_file_path = BASEDIR / "outputs/para_sets/coqa_score_set.json"
    result_set_file_path = BASEDIR / "datasets/CoQA/Evaluation/coqa_para.json"

    dataset = _load_data(data_set_file_path)
    result_data = _load_data(result_set_file_path)

    items = _filter_items_by_parameter_group(dataset, 'CoQA', qdg)
    i = 0
    for story in result_data['data']:
        for question in story['questions']:
            question['input_text'] = items[i]
            print(f'Working on: {items[i]}')
            i += 1

    _save_data(result_set_file_path, result_data)


def create_com2sense_para_set(qdg='medium'):
    print("\nCreating COM2SENSE para set")

    data_set_file_path = BASEDIR / "outputs/para_sets/com2sense_score_set.json"
    result_set_file_path = BASEDIR / "datasets/COM2SENSE/Evaluation/COM2SENSE_para.json"

    dataset = _load_data(data_set_file_path)
    result_data = _load_data(result_set_file_path)

    items = _filter_items_by_parameter_group(dataset, 'COM2SENSE', qdg)
    i = 0
    for example in result_data['examples']:
        example['sent'] = items[i]
        print(f'Working on: {items[i]}')
        i += 1

    _save_data(result_set_file_path, result_data)


def create_socialIqa_para_set(qdg='low'):
    print("\nCreating SocialIQA para set")

    data_set_file_path = BASEDIR / "outputs/para_sets/social_iq_score_set.json"
    result_set_file_path = BASEDIR / "datasets/Social_IQa/Evaluation/socialIqa_para.json"

    dataset = _load_data(data_set_file_path)
    result_data = _load_data(result_set_file_path)

    items = _filter_items_by_parameter_group(dataset, 'SocialIQa', qdg)
    i = 0
    for example in result_data['examples']:
        example['input'] = items[i]
        print(f'Working on: {items[i]}')
        i += 1

    _save_data(result_set_file_path, result_data)


if __name__ == "__main__":
    create_socialIqa_para_set()
    create_com2sense_para_set()
    create_coqa_para_set()
