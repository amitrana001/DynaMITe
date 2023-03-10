#!/bin/bash -x

export PYTHONPATH=.:$PYTHONPATH
. /home/rana/anaconda3/etc/profile.d/conda.sh
conda activate m2f


random_port=$((RANDOM + 10000 ))
echo "$random_port"
MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:$random_port"

NCCL_DEBUG=INFO python ablation_multi_insts_final.py --eval-only --config-file $4 --dist-url "$DIST_URL" --num-gpus $3 --eval-dataset $1 --eval-strategy $2 MODEL.WEIGHTS $5
