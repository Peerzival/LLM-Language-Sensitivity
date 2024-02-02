#!/bin/bash
#SBATCH --job-name=paraphrase_dataset
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH -p medium
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=4G
#SBATCH -C scratch
#SBATCH --time=05:00:00

module load python

echo "Current directory is: $(pwd)"
if [ -d "/scratch/users/gnewuch/project" ]; then
  cp -r ~/LLM-Language-Sensitivity /scratch/users/gnewuch/project
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  python -m venv env
  source env/bin/activate
  pip install -r requirements.txt
python3 create_social_iqa_para.py
deactivate