#!/usr/local_rwth/bin/zsh

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --output=./output/eval_%A.out

export PYTHONPATH=.:$PYTHONPATH
. /home/qn313466/anaconda/etc/profile.d/conda.sh
conda activate m2f

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:10088"

NCCL_DEBUG=INFO python iterative_train_net.py --eval-only --config-file $1 --dist-url "$DIST_URL" --num-gpus $3 MODEL.WEIGHTS $2
