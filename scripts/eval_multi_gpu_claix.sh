#!/usr/local_rwth/bin/zsh

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --output=./output/eval_%A.out

export PYTHONPATH=.:$PYTHONPATH
. /home/qn313466/anaconda/etc/profile.d/conda.sh
conda activate m2f

NCCL_DEBUG=INFO python iterative_train_net.py --eval-only --config-file $1 --num-gpus $3 MODEL.WEIGHTS $2 DATALOADER.NUM_WORKERS 0
