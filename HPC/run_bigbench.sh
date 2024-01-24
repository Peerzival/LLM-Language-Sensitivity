#!/bin/bash
#SBATCH --job-name=coqa_gpt_2_eval_cpu
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH -ntasks = 1
#SBATCH -p gpu
#SBATCH --mem-per-gpu=20G
#SBATCH --time=03:00:00
#SBATCH -C scratch

module load python

if [ -d "/scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase" ]; then
    source /scratch/users/gnewuch/env/bin/activate
    python3 ~/llm_paraphrase/test_full_CoQa/task.py
else
    mkdir -p /scratch/users/gnewuch/llm_evaluation_project
    cp -r ~/llm_paraphrase /scratch/users/gnewuch/llm_evaluation_project
    cd /scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase

    python -m venv /scratch/users/gnewuch/env
    source /scratch/users/gnewuch/env/bin/activate
    pip install -r requirements.txt
    pip install git+https://github.com/google/BIG-bench.git
    python3 ~/llm_paraphrase/test_full_CoQa/task.py
fi