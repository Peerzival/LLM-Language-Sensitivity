#!/bin/bash
#SBATCH --job-name=coqa_gpt_2_eval_cpu
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH -G gtx1080
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01:00:00
#SBATCH -C scratch

module load python

echo "Current directory is: $(pwd)"
if [ -d "/scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase" ]; then
    cd /scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase
    source /scratch/users/gnewuch/env/bin/activate
    echo "Current directory is: $(pwd)"
    python3 test_full_CoQa/task.py
else
    mkdir -p /scratch/users/gnewuch/llm_evaluation_project
    cp -r ~/llm_paraphrase /scratch/users/gnewuch/llm_evaluation_project
    cd /scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase

    python -m venv /scratch/users/gnewuch/env
    source /scratch/users/gnewuch/env/bin/activate
    pip install -r requirements.txt
    pip install git+https://github.com/google/BIG-bench.git
    echo "Hello Max"
    python3 /llm_paraphrase/test_full_CoQa/task.py
fi
