# -*- coding: utf-8 -*-
""" Implementation of Vision Transformer. 
    Note that there are drop paths in the block. 
"""

import torch
import torch.nn as nn
from model.Transformer import EncoderBlock
from model.embed import PatchEmbedding, PositionalEncoding

class VisionTransformer(nn.Module): 
    """ Vision Transformer. 
        Learn from timm. 
        
        Usually d_k * n_heads = d_model, d_ff = 4 * d_model. 
        Note that default qkv_bias is False in this implement, but it is True in timm. 
    """
    def __init__(self, input_resolution=224, patch_size=16, channel=3, layers_num=12, 
                 d_model=768, d_ff=3072, n_heads=12, d_k=64, d_v=64, attn_drop=0., proj_drop=0., 
                 drop_path=0., qkv_bias=False, layer_norm_eps=1e-5, hidden_act='relu', 
                 pos_embed='learnable', feature_process='cls', num_classes=0): 
        """
            input_resolution: input image resolution
            patch_size: the edge size of each patch
            channel: image channel, 3 for colorful image
            layers_num: indentical encoder layers
            d_model: dimension of each word
            d_ff: the dimension of the inner-layer in FFN
            n_heads: number of multi-heads
            d_k: dimension of key, which is same to the dimension of query
            d_v: dimension of value
            attn_drop: 0-1, 0 means without attention dropout
            proj_drop: 0-1, 0 means without linear dropout
            drop_path: 0-1, 0 means without drop path
            qkv_bias: whether linear for getting Q, K, V has bias
            layer_norm_eps: eps for nn.LayerNorm
            hidden_act: activation in FFN, use as 'relu', 'leakyrelu', 'gelu', 'quickgelu'
            pos_embed: '1d' and '2d': fixed sin-cos embedding; 'learnable': 1d learnable embedding; 'none': no embedding
            feature_process: feature_process for the MLP head, 'none' for output features
            num_classes: num_classes for MLP head, 0 for None module
        """

        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(input_resolution, patch_size, channel, d_model)

        #a learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=.02)

        #positional embedding
        #2d is same as 1d
        assert pos_embed in ('1d', '2d', 'learnable', 'none'), 'Could not use ' + str(pos_embed) + ' embedding. '
        if pos_embed == '1d': 
            self.register_buffer('pos_embed', PositionalEncoding.get_1d_absolute_sin_cos_embedding((input_resolution // patch_size) ** 2, d_model))
        elif pos_embed == '2d': 
            self.register_buffer('pos_embed', PositionalEncoding.get_2d_absolute_sin_cos_embedding(input_resolution // patch_size, input_resolution // patch_size, d_model))
        elif pos_embed == 'learnable': 
            self.pos_embed = PositionalEncoding.get_1d_learnable_embedding((input_resolution // patch_size) ** 2, d_model)
            nn.init.normal_(self.pos_embed, std=.02)
        #add more if you want
        else: 
            self.pos_embed = None#for special task
        
        self.pos_drop = nn.Dropout(p=proj_drop) if proj_drop > 0 else nn.Identity()

        #block
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path, layers_num)]
        self.blocks = nn.Sequential(*[VisionTransformerBlock(d_model, d_ff, n_heads, d_k, d_v, attn_drop, proj_drop, drop_path_rate[i], qkv_bias, layer_norm_eps, hidden_act) for i in range(layers_num)])

        assert feature_process in ('none', 'cls', 'mean'), 'Could only use "none", "cls" or "mean". '
        self.feature_process = feature_process
        
        #final norm
        self.encode_norm = nn.Identity() if self.feature_process == 'mean' else nn.LayerNorm(d_model)
        self.fc_norm = nn.LayerNorm(d_model) if self.feature_process == 'mean' else nn.Identity()

        #feature process, none for outputing raw features
        assert not (feature_process == 'none' and num_classes > 0), 'Parameter "num_classes" must be 0 when "feature_process" is "none". '
        if num_classes > 0: 
            self.mlp_head = nn.Linear(d_model, num_classes)
            nn.init.normal_(self.mlp_head.weight, std=.02)#std=2e-5 in MAE source codes
        else: 
            self.mlp_head = nn.Identity()
        
    
    def forward(self, x): 
        """ x: input images, float32 tensor with size (b, c, h, w)
        """

        x = self.patch_embed(x)

        x = torch.cat((self.cls_token.repeat(x.size(0), 1, 1), x), dim=1)
        
        if self.pos_embed is not None: 
            x[:, 1:, :] += self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        #to get the features of the encoder ,use feature_process='none' and num_classes=0
        #features process
        x = self.encode_norm(x)
        if self.feature_process == 'mean': 
            #learn from MAE
            x = x[:, 1:, :].mean(dim=1)
        elif self.feature_process == 'cls': 
            x = x[:, 0, :]
        #else: None
        x = self.fc_norm(x)

        x = self.mlp_head(x)

        return x



class VisionTransformerBlock(EncoderBlock): 
    """ Independent ViT encoder block. 
        Contains self-attention and feed forward network. 
        Note that in the normalization layer is located before each sublayer. 

        Add drop path to EncoderBlock, and remove mask in forward. 
    """

    def __init__(self, d_model=512, d_ff=2048, n_heads=8, d_k=64, d_v=64, attn_drop=0., 
                 proj_drop=0., drop_path=0., qkv_bias=False, layer_norm_eps=1e-5, hidden_act='relu'): 
        
        super(VisionTransformerBlock, self).__init__(d_model, d_ff, n_heads, d_k, d_v, attn_drop, 
                                                     proj_drop, qkv_bias, layer_norm_eps, hidden_act)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, enc_inputs): 
        norm_inputs = self.layernorm1(enc_inputs)#layer normalization is the first
        outputs =  self.enc_self_attn(norm_inputs, norm_inputs, norm_inputs)

        tmp = enc_inputs + self.drop_path(outputs)#with drop path
        
        norm_inputs =  self.layernorm2(tmp)
        outputs =  self.enc_ffn(norm_inputs)

        outputs = tmp + self.drop_path(outputs)#with drop path

        return outputs



class DropPath(nn.Module): 
    """ Drop paths (Stochastic Depth) per sample (When applied in main path of residual blocks). 
        Learn from timm. 
    """

    def __init__(self, drop_prob=0.): 
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x): 
        if self.drop_prob <= 0 or not self.training: 
            return x
        
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (torch.rand(shape, dtype=x.dtype, device=x.device) >= self.drop_prob).float()

        output = x / (1 - self.drop_prob) * random_tensor
        return output
