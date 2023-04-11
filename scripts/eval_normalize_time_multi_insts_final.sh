#!/bin/bash -x

export PYTHONPATH=.:$PYTHONPATH
. /home/rana/anaconda3/etc/profile.d/conda.sh
conda activate m2f

PORT=$(shuf -n 1 -i 49152-65535)
MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:$PORT"

NCCL_DEBUG=INFO python iterative_eval_multi_inst.py --eval-only --config-file $1 --normalize-time --eval-dataset $4 --eval-strategy $5 --seed-id $6 --dist-url "$DIST_URL" --num-gpus $3 MODEL.WEIGHTS $2
