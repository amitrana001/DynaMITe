#!/bin/bash -x

#SBATCH --account=objectsegvideo

#SBATCH --partition=booster

#SBATCH --nodes=4

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=48

#SBATCH --job-name=dynamite_1024

#SBATCH --gres=gpu:4

#SBATCH --time=20:00:00

#SBATCH --array 0-9%1

#SBATCH --output=/p/project/objectsegvideo/amit/DynaMITe/output/dynamite_coco_4gpus_1024_%A_%a.log


echo "Activating conda env and project..."
jutil env activate -p objectsegvideo

source /p/project/objectsegvideo/sabari/libs/anaconda3/bin/activate m2f
module load Stages/2022
module load CUDA/11.5
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5

export OMP_NUM_THREADS=1

## Parse nodelist for multi-node training. Nominate the first node in the $SLURM_JOB_NODELIST as the 'rdzv_endpoint'
PARSED_HOSTLIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
DDP_HOST=${PARSED_HOSTLIST:0:7}

echo "DDP Master Host: ${DDP_HOST}"
echo "Number of nodes: ${SLURM_NNODES}"

## Copy the code repository to a temporary directory so that when the job restarts after 24hrs it is unaffected by underlying changes in the code-base

#REPO_DIR="${PROJECT_objectsegvideo}/ali/code_repos/universal-video-segmentor"
#TMP_REPOS_DIR="${PROJECT_objectsegvideo}/ali/tmp/UVS_code_repos"
#TMP_REPO_COPY_DIR=${TMP_REPOS_DIR}/$(date '+%d.%m.%Y-%H.%M')_${SLURM_JOB_ID}
#mkdir -p ${TMP_REPO_COPY_DIR}
#
#echo "Copying ${REPO_DIR} to ${TMP_REPO_COPY_DIR}"
#rsync -az ${REPO_DIR}/ ${TMP_REPO_COPY_DIR}/
#cd ${TMP_REPO_COPY_DIR}

## Add the temporary code directory to $PYTHONPATH
#export PYTHONPATH=${TMP_REPO_COPY_DIR}:$PYTHONPATH
cd ..
export PYTHONPATH=.:$PYTHONPATH
SOCKET_NAME=$(ip r | grep default | awk '{print $5}')
export NCCL_SOCKET_IFNAME=$SOCKET_NAME

## Main training command
#srun --unbuffered torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=22021994 --rdzv_backend=c10d --rdzv_endpoint ${DDP_HOST}i.juwels train_net.py --config-file $1 
#srun --unbuffered torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=22021994 --rdzv_backend=c10d --rdzv_endpoint ${DDP_HOST}.juwels train_net.py --config-file $1 
#srun --unbuffered torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=22021994 --rdzv_backend=c10d --rdzv_endpoint ${DDP_HOST} train_net.py --config-file $1 
#srun --unbuffered torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=22021994 --rdzv_backend=c10d --rdzv_endpoint ${DDP_HOST}.juwels train_net.py --config-file $1  --num-gpus $SLURM_GPUS --num-machines $SLURM_NNODES
mkdir -p output/$2

#NCCL_DEBUG=INFO srun --unbuffered torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=22021994 --rdzv_backend=c10d --rdzv_endpoint ${DDP_HOST}i.juwels iterative_train_net.py --config-file $1 --num-gpus 4 --num-machines 4 --resume SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 1e-4 OUTPUT_DIR output/$2 SOLVER.CHECKPOINT_PERIOD 10000
NCCL_DEBUG=INFO srun --unbuffered torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=22021994 --rdzv_backend=c10d --rdzv_endpoint ${DDP_HOST}i.juwels iterative_train_net.py --config-file $1 --num-gpus 4 --num-machines 4 --resume OUTPUT_DIR output/$2 SOLVER.CHECKPOINT_PERIOD 10000
