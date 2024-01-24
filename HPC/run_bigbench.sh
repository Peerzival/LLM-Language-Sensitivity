#!/bin/bash
#SBATCH --job-name=coqa_gpt_2_eval_cpu
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH -p medium
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH -C scratch

module load python

echo "Current directory is: $(pwd)"
if [ -d "/scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase" ]; then
    cd /scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase
    source env/bin/activate
    echo "Current directory is: $(pwd)"
    python3 test_full_CoQa/task.py
else
    mkdir -p /scratch/users/gnewuch/llm_evaluation_project
    cp -r ~/llm_paraphrase /scratch/users/gnewuch/llm_evaluation_project
    cd /scratch/users/gnewuch/llm_evaluation_project/llm_paraphrase
    python -m venv /scratch/users/gnewuch/env
    source /scratch/users/gnewuch/env/bin/activate
