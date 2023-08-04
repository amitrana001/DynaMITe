#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=./output/eval_%A.out

export PYTHONPATH=.:$PYTHONPATH
. /home/rana/anaconda3/etc/profile.d/conda.sh
conda activate m2f

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:17708"

NCCL_DEBUG=INFO python iterative_eval_single_inst.py --eval-only --config-file $1  --dist-url "$DIST_URL" --num-gpus 1 MODEL.WEIGHTS $2
