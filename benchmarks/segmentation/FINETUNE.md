## ADE20K Semantic Segmentation

Once pretrained, you can an UPerNet model for semantic segmentation using [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.2), similar to the examples provided [here](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.2/configs/mae).

### Step 1: Install MMSegmentation v1.1.2

First, install `mmsegmentation` version 1.1.2:
```
pip install "mmsegmentation==1.1.2"
```

Verify the installation following the steps specified [here](https://mmsegmentation.readthedocs.io/en/main/get_started.html).

### Step 2: Convert the Pretrained Model

Next, convert the pretrained model using the provided script:

```
python benchmarks/segmentation/tools/model_converters/mmpre2mmseg.py {input_checkpoint_path} {output_checkpoint_path}
```
You can download our pretrained ColorMAE models from [here](../../README.md#pretrained-models).

For example, if you're using the `colormae-green-epoch_300.pth` checkpoint, run the following command:
```
python benchmarks/segmentation/tools/model_converters/mmpre2mmseg.py pretrained/colormae-green-epoch_300.pth pretrained/colormae-green-epoch_300_converted_mmseg.pth
```

### Step 3: Fine-Tune the Model

Finally, run the `benchmarks/segmentation/tools/train.py` file with `torchrun`, specifying the necessary options for fine-tuning:

#### Example Command

```
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    benchmarks/segmentation/tools/train.py benchmarks/segmentation/configs/colormae-base_upernet_8xb2-amp-160k_ade20k-512x512.py \
    --launcher pytorch \
    --resume \
    --work-dir "./work_dirs/colormae-300e-G/finetune/segmentation" \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=<your_converted_checkpoint> \
    model.backbone.init_cfg.prefix="backbone." \
    model.pretrained=None \
    train_dataloader.batch_size=4 \
```
here `model.backbone.init_cfg.checkpoint=<your_converted_checkpoint>` specifies the path to the converted checkpoint in Step 2; for example `pretrained/colormae-green-epoch_300_converted_mmseg.pth`.

See [`benchmarks/segmentation/tools/examples/segmentation_colormae_green_300e.slurm`](tools/examples/segmentation_colormae_green_300e.slurm) for a full example.

Please refer to the [Semantic Segmentation on ADE20K](../../README.md#semantic-segmentation-on-ade20k) section in the `README.md` for model checkpoints, logs, and the results reported in the paper.