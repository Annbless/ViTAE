import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
import numpy as np
from .vitmodules import ViTAE_ViT_basic

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViTAE_basic_7': _cfg(),
}

#  The tiny model
@register_model
def ViTAE_basic_7(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 128], token_dims=[64, 64, 256], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 7], NC_heads=[1, 1, 4], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The small model
@register_model
def ViTAE_basic_14(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 192], token_dims=[64, 64, 384], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 14], NC_heads=[1, 1, 6], RC_heads=[1, 1, 1], mlp_ratio=3., NC_group=[1, 1, 96], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The 6M model
@register_model
def ViTAE_basic_10(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 128], token_dims=[64, 64, 256], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 10], NC_heads=[1, 1, 4], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The 13M model
@register_model
def ViTAE_basic_11(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 160], token_dims=[64, 64, 320], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 11], NC_heads=[1, 1, 5], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model