# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_mae(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('backbone.patch_embed'):
            new_key = k.replace('backbone.patch_embed.projection', 'patch_embed.proj')
            new_ckpt[new_key] = v
            continue
        if k == 'backbone.ln1.weight':
            new_key = 'norm.weight'
            new_ckpt[new_key] = v
            continue
        if k == 'backbone.ln1.bias':
            new_key = 'norm.bias'
            new_ckpt[new_key] = v
            continue
        if k.startswith('data_preprocessor'):
            continue
        if k.startswith('neck'):
            continue
        if k.startswith('backbone.'):
            if k.startswith('backbone.layers'):
                new_key = k.replace('backbone.layers.','blocks.')
            else:
                new_key = k.replace('backbone.','')

            if "ln" in new_key:
                new_key = new_key.replace("ln","norm")
            if "ffn.layers.0.0." in new_key:
                new_key = new_key.replace("ffn.layers.0.0.","mlp.fc1.")
            if "ffn.layers.1." in new_key:
                new_key = new_key.replace("ffn.layers.1.","mlp.fc2.")
            new_ckpt[new_key] = v
        else:
            new_key = k
            new_ckpt[new_key] = v

    return {'model': new_ckpt}


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained beit models to'
        'MMDetection style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_mae(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
