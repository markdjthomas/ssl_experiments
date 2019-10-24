#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --account=def-stanmat-ab
#SBATCH --time=0-23:59
#SBATCH --output=logs/tensorboard-%N-%j.out
#SBATCH --mail-user=<markthomas@dal.ca>
#SBATCH --mail-type=ALL

module load nixpkgs/16.09 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 python/3.6
source ~/pytorch/bin/activate

tensorboard --logdir logs/tensorboard --host 0.0.0.0 > logs/tensorboard.out
