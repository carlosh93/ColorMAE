# Directly inherit the entire recipe you want to use.
_base_ = 'mmpretrain::mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'
# This line is to import your own modules.
custom_imports = dict(imports='models')

# Modify the backbone to use your own backbone.
_base_['model']['backbone'] = dict(type='ColorMAEViT', arch='b', patch_size=16, mask_ratio=0.75)

# Select the masking strategy
_base_['model']['backbone']['mask_type'] = dict(
    begin=dict(
        name="green",
        data_path="noise_colors/green/green_noise_data_3072.npz"))


# Also, you can "mix" multiple color patterns in the training process.
# Uncomment the following lines to use multiple color patterns.
# _base_['model']['backbone']['mask_type'] = dict(
#     begin=dict(
#         name="green",
#         data_path="noise_colors/green/green_noise_data_3072.npz"),
#     epoch_200=dict(
#         name="blue",
#         data_path="noise_colors/blue/blue_noise_data_3072.npz"),
#     epoch_250=dict(
#         name="red",
#         data_path="noise_colors/red/red_noise_data_3072.npz"),
#     epoch_275=dict(
#         name="purple",
#         data_path="noise_colors/purple/purple_noise_data_3072.npz"))


# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size. Check: https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py#L56
# This is the default auto_scale_lr setting when training 8 GPUs
# Change it if you use different number of GPUs.
# _base_['auto_scale_lr'] = dict(base_batch_size=4096)  # effective batch size = 512*8 = 4096