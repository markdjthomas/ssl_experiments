#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lgpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=def-stanmat-ab
#SBATCH --time=1-23:59
#SBATCH --output=logs/jupyter-%N-%j.out
#SBATCH --mail-user=<markthomas@dal.ca>
#SBATCH --mail-type=ALL

module load nixpkgs/16.09 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 python/3.6
source ~/pytorch/bin/activate

bash $VIRTUAL_ENV/bin/notebook.sh > logs/jupyter_notebook.out
