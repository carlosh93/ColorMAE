_base_ = [
    './_base_/models/upernet_mae.py', './_base_/datasets/ade20k.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/mae_pretrain_vit_large_mmcls.pth',  # Update this path
    backbone=dict(
        type='MAE',
        img_size=(512, 512),
        patch_size=16,
        embed_dims=1024,  # Changed to 1024 for ViT-Large
        num_layers=24,  # Changed to 24 for ViT-Large
        num_heads=16,  # Changed to 16 for ViT-Large
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.2,  # 0.1
        out_indices=[7, 11, 15, 23]),  # [3, 5, 7, 11]
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),  # Changed embed_dim to 1024
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],  # Updated to 1024 for all layers
        num_classes=150,
        channels=1024),  # Changed channels to 1024
    auxiliary_head=dict(in_channels=1024, num_classes=150),  # Updated in_channels to 1024
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=9e-5, betas=(0.9, 0.999), weight_decay=0.05), # 1e-4 -> 9e-5
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.75),  # 0.65 -> 0.75
    constructor='LayerDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000, max_keep_ckpts=3))