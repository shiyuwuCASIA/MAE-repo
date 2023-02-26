# -*- coding: utf-8 -*-
""" Implementation of MAE. 
""" 

import torch
import torch.nn as nn
from model.Transformer import TransformerEncoder
from model.VisionTransformer import VisionTransformer
from model.embed import PositionalEncoding

class MaskedAutoEncoder(VisionTransformer): 
    """ Implement of MAE. 
        
        Module only used when pretraining. 
        The encoder parameters are same as our VisionTransformer, so it's easy to finetuning using the checkpoint. 
    """

    def __init__(self, input_resolution=224, patch_size=16, channel=3, layers_num=12, d_model=768, 
                 d_ff=3072, n_heads=12, d_k=64, d_v=64, qkv_bias=False, layer_norm_eps=1e-5, 
                 hidden_act='relu'): 
        """ No dropout and drop path during pretraining. 
            Set qkv_bias == True. 
            During training, use two 2d sin-cos positional embedding. 
        """

        super(MaskedAutoEncoder, self).__init__(input_resolution, patch_size, channel, layers_num, d_model, 
                                                d_ff, n_heads, d_k, d_v, 0., 0., 0., qkv_bias, layer_norm_eps, 
                                                hidden_act, '2d', 'none', 0)
        """ Attributes patch_embed, pos_embed, blocks, encode_norm are from ViT. 
        """
        #we use a fixed decoder
        #all params' names start with 'decoder'
        self.decoder_embed = nn.Linear(d_model, 512)

        self.decoder = TransformerEncoder(8, 512, 2048, 16, 32, 32, 0., 0., qkv_bias, layer_norm_eps, hidden_act)
        
        #2d sin-cos embedding as MAE
        self.register_buffer('decoder_pos_embed', PositionalEncoding.get_2d_absolute_sin_cos_embedding(input_resolution // patch_size, input_resolution // patch_size, 512))

        self.decoder_pred = nn.Sequential(nn.LayerNorm(512), 
                                          nn.Linear(512, patch_size ** 2 * channel))
        
        self.decoder_mask_token = nn.Parameter(torch.zeros(1, 1, 512))
        nn.init.normal_(self.decoder_mask_token, std=.02)
    
    def forward(self, x, mask_ratio=0.75): 
        """ x: input images, float32 tensor with size (b, c, h, w)
        """

        x = self.patch_embed(x)

        #add positional embedding, no cls_token here
        if self.pos_embed is not None: 
            x += self.pos_embed

        #drop those patches in the end
        idx = torch.stack([torch.randperm(x.size(1), device=x.device) for _ in range(x.size(0))])
        idx_restore = torch.argsort(idx, dim=1)
        keep_len = int(x.size(1) * (1 - mask_ratio))
        idx_keep = idx[:, :keep_len]

        #generate a mask for calculating loss
        mask = torch.ones(x.size(), device=x.device)
        mask[:, :keep_len, :] = 0
        pixel_mask = torch.gather(mask, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, x.size(2)))

        #get those visible, unmasked patches
        x = torch.gather(x, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))
        x = torch.cat((self.cls_token.repeat(x.size(0), 1, 1), x), dim=1)

        x = self.blocks(x)
        x = self.encode_norm(x)

        x = self.decoder_embed(x)

        #restore the input of the decoder
        mask_tokens = self.decoder_mask_token.repeat(x.size(0), idx_restore.size(1) - keep_len, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, x.size(2)))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x[:, 1:, :] += self.decoder_pos_embed

        x = self.decoder(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]#remove cls
        #x: (B, L, D)
        
        return x, pixel_mask
