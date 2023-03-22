#!/bin/bash -x

export PYTHONPATH=.:$PYTHONPATH
. /home/rana/anaconda3/etc/profile.d/conda.sh
conda activate m2f

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:10009"

NCCL_DEBUG=INFO python iterative_eval_single_inst.py --eval-only --config-file $1 --dist-url "$DIST_URL" --num-gpus $3 MODEL.WEIGHTS $2
