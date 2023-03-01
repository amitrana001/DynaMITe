#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
. /home/rana/anaconda3/etc/profile.d/conda.sh
conda activate m2f

NCCL_DEBUG=INFO python iterative_train_net.py --eval-only --config-file $1 --num-gpus $3 MODEL.WEIGHTS $2
