# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import urllib
import timm.models.vision_transformer
import os

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

URL_MODELS = {
    "mae-base": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
    "mae-large": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
}

def get_mae_vit(model_name):
    if 'large' in model_name:
        model = mae_large_patch16()
    elif 'base' in model_name:
        model = mae_base_patch16()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    load_pretrained_weights_mae(model, model_name = model_name)
    return model

def load_pretrained_weights_mae(model, model_name=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_weights = os.path.join(current_dir, f"backbone_checkpoints/{model_name}.pth")
    if os.path.isfile(pretrained_weights):
        print(f"Load pre-trained checkpoint from: {pretrained_weights}")
    else:
        print(f"Not found pretrained weights for {model_name}, downloading from {URL_MODELS[model_name]} and save to {pretrained_weights}")
        assert model_name in URL_MODELS, f"Model name {model_name} not in URL_MODELS"
        url = URL_MODELS[model_name]
        pretrained_weights = os.path.join(current_dir, "backbone_checkpoints", f"{model_name}.pth")
        os.makedirs(os.path.dirname(pretrained_weights), exist_ok=True)
        urllib.request.urlretrieve(url, pretrained_weights)
    checkpoint_model = torch.load(pretrained_weights, map_location="cpu")['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return


def mae_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    return model

