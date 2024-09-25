#!/bin/bash

hostname
nvidia-smi
pwd

# Enable debugging mode
set -x

# define some environment variables
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/pretrain/venv_pre/

# #run the application:
# module purge
# #module load gcc/6.4.0
# module load cuda/11.8
echo $ENV_PREFIX
source ~/.miniconda3/etc/profile.d/conda.sh # check your conda path
conda activate "$ENV_PREFIX"
echo "Runing python from ..."
which python

export DECORD_EOF_RETRY_MAX=20480
# export Wandb KEY
# export WANDB_API_KEY=638f3ee09bb5b0ae79def2434f17c07e20c5c3a0

# Get the IP address and set port for MASTER node
echo "NODELIST="${SLURM_NODELIST}
master_ip=localhost
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

# command to submit the next job
command="unset+SLURM_CPU_BIND+&&+sbatch+pretrain/run.slurm+$1+$2+$3"
echo command=$command

# Print the number of GPUs allocated per node
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS=$SLURM_JOB_NUM_GPUS
export CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export PORT=$master_port
export MASTER_ADDR=$master_ip
export PYTHONPATH="$PWD":$PYTHONPATH

echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Number of nodes allocated: $NUM_NODES"
echo "Total number of GPUs allocated: $GPUS"
echo "Total number of CPUs allocated: $CPUS_PER_TASK"


# Checks
if [ -z "$1" ] || [ "$1" = "None" ]; then
    echo "Error: Config file is missing or set to 'None'"
    exit 1
fi

if [ -z "$2" ] || [ "$2" = "None" ]; then
    echo "Error: --slurm_epochs argument is missing or set to 'None'"
    exit 1
fi

if [ -z "$3" ] || [ "$3" = "None" ]; then
  echo "Error: --work-dir argument is missing or set to 'None'"
  exit 1
fi

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_ip:$master_port \
    tools/train.py $1 \
    --launcher pytorch \
    --slurm_command $command \
    --slurm_epochs $2 \
    --work-dir $3

# 4h -> 50 epochs
# 24 -> 300 epochs
# 72 -> 900 epochs