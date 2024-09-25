Under Construction... 🚧

<!-- # How to run
# in root dir:
# sbatch run_slurm/run_finetune_segmentation_A100.slurm benchmarks/mmsegmentation/configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512.py {checkpoint} {work-dir}

# currently running: sbatch run_slurm/run_finetune_segmentation_A100.slurm benchmarks/mmsegmentation/configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512.py work_dirs/random_baseline/pretrain/epoch_300_converted_mmcls.pth "./work_dirs/random_baseline/finetune_segmentation"

# debug: python benchmarks/mmsegmentation/tools/train.py benchmarks/mmsegmentation/configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512.py --cfg-options model.backbone.init_cfg.type=Pretrained model.backbone.init_cfg.checkpoint=work_dirs/random_baseline/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k/epoch_300.pth model.backbone.init_cfg.prefix="backbone." model.pretrained=None --work-dir "./debug"

# Before Running: Convert checkpoint
# python benchmarks/mmsegmentation/tools/model_converters/mmpre2mmseg.py {input_checkpoint_path} {output_checkpoint_path}_prop_blueNoise_{exp#}_converted_mmcls.pth -->