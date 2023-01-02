#!/usr/local_rwth/bin/zsh

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --mem=50G
#SBATCH --time=4-00:00:00
#SBATCH --output=./output/multi_queries/mq_%A.out

export PYTHONPATH=.:$PYTHONPATH
. /home/qn313466/anaconda/etc/profile.d/conda.sh
conda activate m2f

if [ "$#" -eq 2 ]
then 	
	NCCL_DEBUG=INFO python train_net.py --resume --config-file $1 --num-gpus $2 
elif [ "$#" -eq 3 ]
then 	
	NCCL_DEBUG=INFO python train_net.py --num-gpus $1 --config-file $2 SOLVER.IMS_PER_BATCH $3
elif [ "$#" -eq 5 ]
then
	NCCL_DEBUG=INFO python iterative_train_net.py --config-file $1 --num-gpus $2 --num-machines 1 OUTPUT_DIR "$5" SOLVER.IMS_PER_BATCH $3 SOLVER.BASE_LR $4 DATALOADER.NUM_WORKERS 3
elif [ "$#" -eq 6 ]
then
	NCCL_DEBUG=INFO python iterative_train_net.py --config-file $1 --machine-rank $5 --dist-url $6 --num-machines 2 --num-gpus $2 SOLVER.IMS_PER_BATCH $3 SOLVER.BASE_LR $4
fi