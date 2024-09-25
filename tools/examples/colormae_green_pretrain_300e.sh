#!/bin/bash

hostname
nvidia-smi
pwd

# Enable debugging mode
set -x

# define some environment variables
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/venv

# Add the current folder to PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH

# run the application:

echo $ENV_PREFIX
source ~/.miniconda3/etc/profile.d/conda.sh # check your conda path
conda activate "$ENV_PREFIX"
echo "Runing python from ..."
which python

export DECORD_EOF_RETRY_MAX=20480

# Get the IP address and set port for MASTER node
echo "NODELIST="${SLURM_NODELIST}
master_ip=localhost
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

# command to submit the next job
command="unset+SLURM_CPU_BIND+&&+sbatch+tools/examples/colormae_green_pretrain_300e.sh+$1+$2+$3"
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
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_ip:$master_port \
    tools/train.py $1 \
    --run_slurm_epochs \
    --slurm_command $command \
    --slurm_epochs $2 \
    --launcher pytorch \
    --work-dir $3

# How to Run This Example:
# sbatch tools/examples/colormae_green_pretrain_300e.sh configs/colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py 100 "./work_dirs/colormae-300e-G/pretrain"
# Make sure work_dirs/colormae-300e-G/pretrain exists