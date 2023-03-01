#!/usr/local_rwth/bin/zsh

#SBATCH --nodes=1
#SBATCH --gres=gpu:16
#SBATCH --gpus-per-node=16
#SBATCH --mem=50G
#SBATCH --time=4-00:00:00
#SBATCH --output=./output/train_%A.out

export PYTHONPATH=.:$PYTHONPATH
. /home/qn313466/anaconda/etc/profile.d/conda.sh
conda activate m2f

NCCL_DEBUG=INFO python iterative_train_net.py --resume --config-file $1 --num-gpus $2 OUTPUT_DIR $3
