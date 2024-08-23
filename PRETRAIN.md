Under Construction... 🚧

### Training commands

**To train with single GPU:**

```bash
# mim train mmpretrain configs/examplenet_8xb32_in1k.py
python tools/train.py configs/colormae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py
```

<!-- **To train with multiple GPUs (1 node 4 gpus):**
```bash
mim train mmpretrain configs/examplenet_8xb32_in1k.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmpretrain configs/examplenet_8xb32_in1k.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmpretrain configs/examplenet_8xb32_in1k.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmpretrain configs/examplenet_8xb32_in1k.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmpretrain configs/examplenet_8xb32_in1k.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results

|       Model        |   Pretrain   | Top-1 (%) | Top-5 (%) |                 Config                  |                Download                |
| :----------------: | :----------: | :-------: | :-------: | :-------------------------------------: | :------------------------------------: |
|  ExampleNet-tiny   | From scratch |   82.33   |   96.15   | [config](./mvitv2-tiny_8xb256_in1k.py)  | [model](MODEL-LINK) \| [log](LOG-LINK) |
| ExampleNet-small\* | From scratch |   83.63   |   96.51   | [config](./mvitv2-small_8xb256_in1k.py) |          [model](MODEL-LINK)           |
| ExampleNet-base\*  | From scratch |   84.34   |   96.86   | [config](./mvitv2-base_8xb256_in1k.py)  |          [model](MODEL-LINK)           |

*Models with * are converted from the [official repo](REPO-LINK). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

<!-- Replace to the citation of the paper your project refers to. -->