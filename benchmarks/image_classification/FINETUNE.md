## ImageNet-1K Classification

Once Pretrained, you can fine-tune the ViT model for Image classification using the `tools/train.py` file and by specifying the appropriate config file located in the `benchmarks/image_classification/configs/` folder. You also need to specify the pretrained ViT model (pre-trained with ColorMAE) using the following parameter `--cfg-options model.backbone.init_cfg.checkpoint`.


For example:

```
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py benchmarks/image_classification/configs/vit-base-p16_8xb128-coslr-100e_in1k.py \
    --launcher pytorch \
    --cfg-options model.backbone.init_cfg.checkpoint=<path_to_pretrained_checkpoint> \
    --resume \
    --work-dir <save_dir>
```

In this example, <path_to_pretrained_checkpoint> should be replaced with the path to your pretrained checkpoint, and <save_dir> is the directory where logs and checkpoints will be saved.

Please refer to the [Image Classification on ImageNet-1k](../../README.md#image-classification-on-imagenet-1k) section in the `README.md` for model checkpoints, logs, and the results reported in the paper.