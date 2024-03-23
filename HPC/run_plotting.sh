#!/bin/bash
#SBATCH --job-name=paraphrase_dataset
#SBATCH --output=job-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxmartin.gnewuch@stud.uni-goettingen.de
#SBATCH -p medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -C scratch
#SBATCH --time=30:00:00

module load python

if [ -d "/scratch/users/gnewuch/project/LLM-Language-Sensitivity" ]; then
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  source env/bin/activate
  cd src/services/
  python3 para_score_plotting_service.py
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  deactivate
else
  cp -r ~/LLM-Language-Sensitivity /scratch/users/gnewuch/project
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  python -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  cd src/services/
  python3 para_score_plotting_service.py
  cp -r /scratch/users/gnewuch/project/LLM-Language-Sensitivity/src/final_scores.png ${HOME}/LLM-Language-Sensitivity/src/
  cd /scratch/users/gnewuch/project/LLM-Language-Sensitivity
  deactivate
fi
