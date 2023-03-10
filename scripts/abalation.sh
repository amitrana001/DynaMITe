#!/bin/bash -x

export PYTHONPATH=.:$PYTHONPATH
. /home/rana/anaconda3/etc/profile.d/conda.sh
conda activate m2f

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:11108"

NCCL_DEBUG=INFO python eval_multi_insts_final.py --eval-only --config-file $1 --eval-dataset $4 --eval-strategy $5 --dist-url "$DIST_URL" --num-gpus $3 MODEL.WEIGHTS $2
