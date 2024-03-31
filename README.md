# The Search for the Holy Prompt 

This is a dataset created for the seminar **Selected Topics in Data Science** for the **Master's programme** at the **Georg-August University in Göttingen**.

>The goal is to investigate the effect of *changes in the language* (e.g. syntax,  wording or word structure) used in prompts on *model performance*.

## Installation

Navigate to the project directory and create a virtual environment by running the following command:
```bash
python3 -m venv env
```
Activate the virtual environment by running the following command:
```bash
source env/bin/activate
```

Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

To paraphrase text run: `Paraphasing/paraphrasing_model.py`

## Dataset

There are three different task categories in the data set, each with seven tasks.

**Table of contents**

* Question answering
  * [CoQaA](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/coqa_conversational_question_answering)
* Emotional understanding
  * [COM2SENSE](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/com2sense)
* Common-sense
  * [Social IQa](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/social_iqa)