## COCO Object Detection and Instance Segmentation

Once pretrained, you can fine-tune the ViTDet model for object detection and instance segmentation using [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v3.2.0), similar to the examples provided [here](https://github.com/open-mmlab/mmdetection/tree/v3.2.0/projects/ViTDet).


### Step 1: Install MMDetection v1.1.2
```
mim install mmdet==3.2.0
```
Verify the installation following the steps specified [here](https://mmdetection.readthedocs.io/en/latest/get_started.html).

### Step 2: Convert the Pretrained Model

Next, convert the pretrained model using the provided script:

```
python benchmarks/object_detection/tools/model_converters/mmpre2mmdet.py {input_checkpoint_path} {output_checkpoint_path}
```
You can download our pretrained ColorMAE models from [here](../../README.md#pretrained-models).

For example, if you're using the `colormae-green-epoch_300.pth` checkpoint, run the following command:
```
python benchmarks/object_detection/tools/model_converters/mmpre2mmdet.py pretrained/colormae-green-epoch_300.pth pretrained/colormae-green-epoch_300_converted_mmdet.pth
```

### Step 3: Fine-Tune the Model

Finally, run the `benchmarks/object_detection/tools/train.py` file with `torchrun`, specifying the necessary options for fine-tuning:

```
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    benchmarks/object_detection/tools/train.py benchmarks/object_detection/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
    --launcher pytorch \
    --resume \
    --work-dir "./work_dirs/colormae-300e-G/finetune/detection" \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=<your_converted_checkpoint> \
    train_dataloader.batch_size=16 \

```
here `model.backbone.init_cfg.checkpoint=<your_converted_checkpoint>` specifies the path to the converted checkpoint in Step 2; for example `pretrained/colormae-green-epoch_300_converted_mmdet.pth`.

See [`benchmarks/object_detection/tools/examples/detection_colormae_green_300e.slurm`](tools/examples/detection_colormae_green_300e.slurm) for a full example.

Please refer to the [Object Detection and Instance Segmentation on COCO](../../README.md#object-detection-and-instance-segmentation-on-coco) section in the `README.md` for model checkpoints, logs, and the results reported in the paper.