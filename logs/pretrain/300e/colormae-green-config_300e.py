auto_scale_lr = dict(base_batch_size=4096)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True,
    type='SelfSupDataPreprocessor')
data_root = 'data/imagenet/'
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='b',
        mask_ratio=0.75,
        mask_type='bn_codes',
        patch_size=16,
        type='MAEViT'),
    head=dict(
        loss=dict(criterion='L2', type='PixelReconstructionLoss'),
        norm_pix=True,
        patch_size=16,
        type='MAEPretrainHead'),
    init_cfg=[
        dict(distribution='uniform', layer='Linear', type='Xavier'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    neck=dict(
        decoder_depth=8,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        embed_dim=768,
        in_chans=3,
        mlp_ratio=4.0,
        patch_size=16,
        type='MAEPretrainDecoder'),
    type='MAE')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.95,
        ), lr=0.0024, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            bias=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=40,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=260,
        begin=40,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=True, seed=0)
resume = True
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=512,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/imagenet/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                crop_ratio_range=(
                    0.2,
                    1.0,
                ),
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        crop_ratio_range=(
            0.2,
            1.0,
        ),
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/proposed_design/exp7/pretrain'
