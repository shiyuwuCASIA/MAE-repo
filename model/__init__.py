# -*- coding: utf-8 -*-
""" Model cards for MAE and ViT. 
"""

from model.VisionTransformer import VisionTransformer
from model.MAE import MaskedAutoEncoder

#vit
def vit_base_patch16(**kwargs): 
    model = VisionTransformer(layers_num=12, d_model=768, d_ff=3072, n_heads=12, 
                              d_k=64, d_v=64, attn_drop=0., proj_drop=0., 
                              qkv_bias=True, layer_norm_eps=1e-6, hidden_act='gelu', **kwargs)
    return model


def vit_large_patch16(**kwargs): 
    model = VisionTransformer(layers_num=24, d_model=1024, d_ff=4096, n_heads=16, 
                              d_k=64, d_v=64, attn_drop=0., proj_drop=0., 
                              qkv_bias=True, layer_norm_eps=1e-6, hidden_act='gelu', **kwargs)
    return model


def vit_huge_patch14(**kwargs): 
    model = VisionTransformer(layers_num=32, d_model=1280, d_ff=5120, n_heads=16, 
                              d_k=80, d_v=80, attn_drop=0., proj_drop=0., 
                              qkv_bias=True, layer_norm_eps=1e-6, hidden_act='gelu', **kwargs)
    return model


#mae
def mae_vit_base_patch16(**kwargs): 
    model = MaskedAutoEncoder(layers_num=12, d_model=768, d_ff=3072, n_heads=12,
                              d_k=64, d_v=64, qkv_bias=True, layer_norm_eps=1e-6, 
                              hidden_act='gelu', **kwargs)
    return model


def mae_vit_large_patch16(**kwargs): 
    model = MaskedAutoEncoder(layers_num=24, d_model=1024, d_ff=4096, n_heads=16,
                              d_k=64, d_v=64, qkv_bias=True, layer_norm_eps=1e-6, 
                              hidden_act='gelu', **kwargs)
    return model


def mae_vit_huge_patch14(**kwargs): 
    model = MaskedAutoEncoder(layers_num=32, d_model=1280, d_ff=5120, n_heads=16,
                              d_k=80, d_v=80, qkv_bias=True, layer_norm_eps=1e-6, 
                              hidden_act='gelu', **kwargs)
    return model