#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       5GB
#SBATCH --ntasks    1
#SBATCH --job-name  ABINEF_featurize
#SBATCH --time      5:00:00
#SBATCH --partition=serial

source /home/kjablonk/anaconda3/bin/activate
conda activate ml

run_featurization /work/lsmo/mof_subset_csdmay2019/ABINEF.cif 20190922_features/
