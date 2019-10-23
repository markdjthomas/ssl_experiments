#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --account=def-stanmat-ab
#SBATCH --time=1-11:59
#SBATCH --output=logs/pretrain-%N-%j.out
#SBATCH --mail-user=<markthomas@dal.ca>
#SBATCH --mail-type=ALL

module load nixpkgs/16.09 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 python/3.6
source ~/pytorch/bin/activate

ARCH=resnet101
SPLIT=0
SECONDS=3
FMAX=1024
EPOCHS=50
BATCH_SIZE=128
LEARNING_RATE=0.01
DECAY_RATE=0.1
AMBIENT_PROB=0.25

RUN_NUM=2
RUN_ID="${ARCH}_split_${SPLIT}_with_ambient_${SECONDS}x${FMAX}_${RUN_NUM}"

python3 -u py/pretrain_backbone.py \
	--run-id $RUN_ID \
   	--arch $ARCH \
    	--split $SPLIT \
	--batch-size $BATCH_SIZE \
	--seconds $SECONDS \
	--fmax $FMAX \
	--num-workers 40 \
	--ambient-prob $AMBIENT_PROB \
	--classes 4 \
	--learning-rate $LEARNING_RATE \
	--decay-rate $DECAY_RATE > logs/$RUN_ID.out
