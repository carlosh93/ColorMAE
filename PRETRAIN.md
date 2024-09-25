<!-- Under Construction... 🚧 -->

## Pre-training commands

### Single GPU
To train with single GPU, you can launch the training in `tools/train.py` with the specific config in the `config` folder. For example:

```bash
python tools/train.py configs/colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py
```

### Multiple GPUs
To train with multiple GPUs (both single and multiple nodes), you can use `torchrun` as follows:

```
torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint={master_ip}:{master_port} tools/train.py {config} --launcher pytorch {train_args}
```

- `num_node`: The number of nodes being used. Typically set to 1 if all GPUs are on the same node (machine).
- `num_gpu`:  The number of GPUs to use per node.
- `rdzv_backend`: Backend for rendezvous. c10d is typically used for PyTorch distributed training.
- `rdzv_endpoint`: Address of the rendezvous master node; you have to specify {master_ip} ip and {master_port} port. For single-node training, this can be localhost:0.
- `config`: The path to the configuration file that defines the model, dataset, training parameters, etc.
`--launcher pytorch`: Specifies that PyTorch’s native distributed training backend should be used for multi-GPU training.
- `train_args` Any additional training arguments that should be passed to the `train.py` script.

Here is an example for 1 node and 4 GPUs using the `configs/colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py` config file:
```
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py \
    --launcher pytorch \
    --work-dir "./work_dirs/colormae-300e-G/pretrain"
```
In this example, `--work-dir` is used to specify the directory where logs and model checkpoints will be saved. For a complete list of parameters, refer to the `tools/train.py` file.

See [`tools/examples/colormae_green_pretrain_300e.sh`](tools/examples/colormae_green_pretrain_300e.sh) for a full example.


```bash
mim train mmpretrain configs/examplenet_8xb32_in1k.py --launcher pytorch --gpus 8
```
### Managing Multi-Node Training with SLURM Command Helpers

When using SLURM for multi-node training, especially if you have **time constraints**, you can utilize the following three parameters specified in `tools/train.py` to manage job submissions effectively: --run_slurm_epochs, --slurm_command, and --slurm_epochs.

- `--run_slurm_epochs`: A flag to specify whether to use the `EpochManagerHook` and activate this mechanism.
- `--slurm_command`: Defines the command to run for the next SLURM job. This should include the full command to resume training from the last checkpoint.
- `--slurm_epochs`: Specifies the number of epochs for training.

By using these parameters, you can run pretraining for a specific number of epochs, exit the current job, and automatically submit the next job. The next job will continue training from the last checkpoint using the command specified in `--slurm_command`.

See [`tools/examples/colormae_green_pretrain_300e.slurm`](tools/examples/colormae_green_pretrain_300e.slurm) for a full example. Note that these parameters can also used without slurm, as shown in [this example](tools/examples/colormae_green_pretrain_300e.sh).