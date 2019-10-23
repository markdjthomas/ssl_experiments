#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lgpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=def-stanmat-ab
#SBATCH --time=0-11:59
#SBATCH --output=logs/train-%N-%j.out
#SBATCH --mail-user=<markthomas@dal.ca>
#SBATCH --mail-type=ALL

module load nixpkgs/16.09 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 python/3.6
source ~/pytorch/bin/activate

ARCH=resnet50
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=0.01
DECAY_RATE=0.1
RUN_NUM=0
RUN_ID="${ARCH}_${RUN_NUM}"

python3 -u py/main.py \
	--run-id $RUN_ID \
   	--arch $ARCH \
	--batch-size $BATCH_SIZE \
	--num-workers 24 \
	--learning-rate $LEARNING_RATE \
	--decay-rate $DECAY_RATE > logs/$RUN_ID.out
