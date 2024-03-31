# The Search for the Holy Prompt 

This is a project for the seminar **Selected Topics in Data Science** for the **Master's programme** at the **Georg-August University in Göttingen**.

>The goal is to investigate the effect of *changes in the language* (e.g. syntax,  wording or word structure) used in prompts on *model performance*.

## Project Structure

The project is structured as follows:

- `src/`: Contains the source code for the project. This includes scripts for evaluating different tasks and models.
- `datasets/`: Contains the datasets used for the project. Each dataset is stored in its own directory.
- `results_no_paraphrase/` and `results_paraphrase/`: Contain the results of the evaluations.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Installation

Navigate to the project directory and create a virtual environment by running the following command:
```bash
python3 -m venv env
```
Activate the virtual environment by running the following command:
```bash
source env/bin/activate
```

Install Big-bench by running the following command:
```bash
 pip install git+https://github.com/google/BIG-bench.git
```

Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Usage

To evaluate the models on the original and paraphrased datasets, navigate to the `src/` directory and run the following scripts:

 - `evaluation_com2sense.py`: Evaluates the models on the COM2SENSE dataset.
 - `evaluation_coqa.py`: Evaluates the models on the CoQA dataset.
 - `evaluation_social_iqa.py`: Evaluates the models on the Social IQa dataset.


**Table of contents**

* Question answering
  * [CoQaA](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/coqa_conversational_question_answering)
* Emotional understanding
  * [COM2SENSE](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/com2sense)
* Common-sense
  * [Social IQa](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/social_iqa)