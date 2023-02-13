#!/usr/local_rwth/bin/zsh

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --mem=50G
#SBATCH --time=4-00:00:00
#SBATCH --output=./output/mq_base/mq_%A.out

export PYTHONPATH=.:$PYTHONPATH
. /home/qn313466/anaconda/etc/profile.d/conda.sh
conda activate m2f

NCCL_DEBUG=INFO python iterative_train_net.py --config-file $1 --num-gpus $2 --num-machines 1 OUTPUT_DIR "$5" SOLVER.IMS_PER_BATCH $3 SOLVER.BASE_LR $4 DATALOADER.NUM_WORKERS 3
