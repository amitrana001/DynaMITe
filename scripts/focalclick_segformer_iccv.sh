#!/bin/bash

#export PYTHONPATH=../:../../:~/soft/coco/PythonAPI/:$PYTHONPATH
export PYTHONPATH=../::$PYTHONPATH
. /home/mahadevan/anaconda3.8/etc/profile.d/conda.sh
conda activate mask2former
module load cuda/11.3

cd ../

declare -a datasets=("DAVIS17Val" "DAVIS17Val" "DAVIS17Val" "DAVIS17Val" "DAVIS17Val" "SBD" "SBD" "SBD" "SBD" "SBD" "COCOVal" "COCOVal" "COCOVal" "COCOVal" "COCOVal")
declare -a strategy=("best" "worst" "random" "random" "random" "best" "worst" "random" "random" "random" "best" "worst" "random" "random" "random")
declare -a random_id=(1 1 1 2 3 1 1 1 2 3 1 1 1 2 3)

#for i in `seq 0 $l`
#do
#       echo $i
#       printf "${ckpt[$i]}"
#done

python scripts/evaluate_model.py FocalClick\
      --model_dir=weights\
      --checkpoint=segformer_b0.pth\
      --logs-path=experiments/iccv/segformer_multi/${datasets[${SLURM_ARRAY_TASK_ID}]}/\
      --infer-size=256\
      --datasets=${datasets[${SLURM_ARRAY_TASK_ID}]}\
      --gpus=0\
      --n-clicks=20\
      --target-iou=1.0\
      --thresh=0.5\
      --dynamite_eval\
      --dynamite_eval_strategy=${strategy[${SLURM_ARRAY_TASK_ID}]}\
      --random_id=${random_id[${SLURM_ARRAY_TASK_ID}]}
      #--model_dir=./experiments/focalclick/resnet50_coco/004_3090/checkpoints/\
      #--model_dir=./experiments/focalclick/resnet50_coco/007_bs64_3090_fuse_feats/checkpoints/\
      #--model_dir=./experiments/focalclick/resnet50_cclvis/001_bs64_1080_fuse_feats/checkpoints/
      #--logs-path=experiments/test
      #--vis
      #--target-iou=0.95\


#--datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SPi,DAVIS17Val, COCOVal\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
