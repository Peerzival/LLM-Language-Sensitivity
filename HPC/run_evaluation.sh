#!/bin/bash
#SBATCH --job-name=paraphrase_dataset
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH -p medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -C scratch
#SBATCH --time=24:00:00

module load python

if [ -d "/scratch/users/gnewuch/project/LLM-Language-Sensitivity" ]; then
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  source env/bin/activate
  python3 evaluation.py
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  deactivate
else
  cp -r ~/LLM-Language-Sensitivity /scratch/users/gnewuch/project
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  python -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  pip install git+https://github.com/google/BIG-bench.git 
  python3 evaluation.py
  cp -r /scratch/users/gnewuch/project/LLM-Language-Sensitivity/results ${HOME}/datasets
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  deactivate
fi