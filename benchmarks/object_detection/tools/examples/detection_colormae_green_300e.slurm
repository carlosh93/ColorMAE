#!/bin/bash --login

#SBATCH --job-name detection_300e_green
#SBATCH --time=00:59:00 # 71:59:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH -o slurm_output/colormae-green-300_detection/%A.out
#SBATCH -e slurm_output/colormae-green-300_detection/%A.err


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
conda activate "$ENV_PREFIX"
echo "Runing python from ..."
which python

export DECORD_EOF_RETRY_MAX=20480

# Get the IP address and set port for MASTER node
echo "NODELIST="${SLURM_NODELIST}
master_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

# command to submit the next job
command="unset+SLURM_CPU_BIND+&&+sbatch+benchmarks/object_detection/tools/examples/detection_colormae_green_300e.slurm+$1+$2+$3"
echo command=$command

# Print the number of GPUs allocated per node
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS=$SLURM_JOB_NUM_GPUS
export CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export PORT=$master_port
export MASTER_ADDR=$master_ip

echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Number of nodes allocated: $NUM_NODES"


# Checks
if [ -z "$1" ] || [ "$1" = "None" ]; then
    echo "Error: Config file is missing or set to 'None'"
    exit 1
fi

if [ -z "$2" ] || [ "$2" = "None" ]; then
    echo "Error: Checkpoint is missing or set to 'None'"
    exit 1
fi

if [ -z "$3" ] || [ "$3" = "None" ]; then
    echo "Error: Work-dir is missing or set to 'None'"
    exit 1
fi


srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_ip:$master_port \
    benchmarks/object_detection/tools/train.py $1 \
    --launcher pytorch \
    --resume \
    --work-dir $3 \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$2 \
    train_dataloader.batch_size=16 \

# Effective batchsize = 16 * 4 = 64 (4 GPUs)

# Before Running: Convert checkpoint
# python benchmarks/object_detection/tools/model_converters/mmpre2mmdet.py {input_checkpoint_path} {output_checkpoint_path}

# How to run (Example)
# in root dir:
# sbatch benchmarks/object_detection/tools/examples/detection_colormae_green_300e.slurm benchmarks/object_detection/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py pretrained/colormae-green-epoch_300_converted_mmdet.pth "./work_dirs/colormae-300e-G/finetune/detection"