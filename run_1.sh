#!/bin/bash
# module load pytorch/1.13.1
export DGL_DISABLE_GRAPHBOLT=1
source amd-env/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8
GPUS=$(( NNODES * GPUS_PER_NODE ))

## master addr and port
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
#export CUDA_VISIBLE_DEVICES=3,2,1,0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_NET="AWS Libfabric"

export FI_CXI_RDZV_THRESHOLD=0 
export FI_CXI_RDZV_GET_MIN=0 
export FI_CXI_RDZV_EAGER_SIZE=0

export SCRIPT="python -u main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions 1 \
  --n-epochs 3000 \
  --model gcn \
  --sampling-rate 1.0 \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --inductive \
  --use-pp \
  --backend nccl \
  --parts-per-node $GPUS_PER_NODE \
  --master-addr $MASTER_ADDR \
  --port $MASTER_PORT \
  --fix-seed"

run_cmd="srun -N 1 -n 8 -c 4 --cpu-bind=cores --gpus-per-node=8 env ./runner.sh"

echo $run_cmd
eval $run_cmd

