#!/bin/bash
#SBATCH --job-name=paraphrase_dataset
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH -p medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -C scratch
#SBATCH --time=20:00:00

module load python

if [ -d "/scratch/users/gnewuch/project/LLM-Language-Sensitivity" ]; then
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  source env/bin/activate
  cd Paraphasing
  python3 create_social_iqa_para.py
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  deactivate
else
  cp -r ~/LLM-Language-Sensitivity /scratch/users/gnewuch/project
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  python -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  cd Paraphasing
  python3 create_social_iqa_para.py && python3 create_coqa_para.py && python3 create_com2sense_para.py
  cp -r /scratch/users/gnewuch/project/LLM-Language-Sensitivity/Paraphasing/para_sets ${HOME}/datasets
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  deactivate
fi
