# -*- coding: utf-8 -*-
""" Implementations of the Transformer, TransformerEncoder and TransformerDecoder.
    Layernorm is in the beginning of each layer, which is different from the original Transformer but more common. 
    
    All data here are torch.tensor or torch.cuda.tensor. 
"""

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class Transformer(nn.Module): 
    """ The total network of Transformer. 
        Only the network part. 
        Decoder is same as Encoder. 
        No option parameters should be here. 
    """

    def __init__(self, layers_num=6, d_model=512, d_ff=2048, n_heads=8, 
                 d_k=64, d_v=64, attn_drop=0., proj_drop=0., qkv_bias=False, 
                 layer_norm_eps=1e-5, hidden_act='relu'): 
        """
            layers_num: number of encoder and decoder layers
            d_model: dimension of each word
            d_ff: the dimension of the inner-layer in FFN
            n_heads: number of multi-heads
            d_k: dimension of key, which is same to the dimension of query
            d_v: dimension of value
            attn_drop: 0-1, 0 means without attention dropout
            proj_drop: 0-1, 0 means without linear dropout
            qkv_bias: whether linear for getting Q, K, V has bias
            layer_norm_eps: eps for nn.LayerNorm
            hidden_act: activation in FFN, use as 'relu', 'leakyrelu', 'gelu', 'quickgelu'

            This module does not contain the output linear. 
        """

        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(layers_num, d_model, d_ff, n_heads, d_k, d_v, 
                                          attn_drop, proj_drop, qkv_bias, layer_norm_eps, hidden_act)
        
        self.decoder = TransformerDecoder(layers_num, d_model, d_ff, n_heads, d_k, d_v, 
                                          attn_drop, proj_drop, qkv_bias, layer_norm_eps, hidden_act)
    
    def forward(self, enc_inputs, dec_inputs, enc_self_mask=None, dec_self_mask=None, enc_dec_mask=None): 
        """ Masks are generated before. 
            Used for training. 
        """
        
        #encode
        enc_outputs = self.encoder(enc_inputs, enc_self_mask)

        #decode
        outputs = self.decoder(dec_inputs, enc_outputs, dec_self_mask, enc_dec_mask)
        #notet that the linear layer is outside this module
        return outputs



class TransformerEncoder(nn.Module): 
    """ Encoder module for Transformer. 
        This module can be used as an independent component. 
    """

    def __init__(self, layers_num=6, d_model=512, d_ff=2048, n_heads=8, d_k=64, d_v=64, 
                 attn_drop=0., proj_drop=0., qkv_bias=False, layer_norm_eps=1e-5, hidden_act='relu'): 
        """
            layers_num: indentical encoder layers
            d_model: dimension of each word
            d_ff: the dimension of the inner-layer in FFN
            n_heads: number of multi-heads
            d_k: dimension of key, which is same to the dimension of query
            d_v: dimension of value
            attn_drop: 0-1, 0 means without attention dropout
            proj_drop: 0-1, 0 means without linear dropout
            qkv_bias: whether linear for getting Q, K, V has bias
            layer_norm_eps: eps for nn.LayerNorm
            hidden_act: activation in FFN, use as 'relu', 'leakyrelu', 'gelu', 'quickgelu'
        """

        super(TransformerEncoder, self).__init__()
        self.encoder_list = nn.ModuleList([EncoderBlock(d_model, d_ff, n_heads, d_k, d_v, attn_drop, proj_drop, qkv_bias, layer_norm_eps, hidden_act) for _ in range(layers_num)])#deep copy
    
    def forward(self, enc_inputs, enc_self_mask=None): 
        enc_outputs = enc_inputs
        for enc_layer in self.encoder_list: 
            enc_outputs = enc_layer(enc_outputs, enc_self_mask)
        
        return enc_outputs



class EncoderBlock(nn.Module): 
    """ Independent encoder block. 
        Contains self-attention and feed forward network. 
        Note that the normalization layer is located before each sublayer. 
    """

    def __init__(self, d_model=512, d_ff=2048, n_heads=8, d_k=64, d_v=64, attn_drop=0., 
                 proj_drop=0., qkv_bias=False, layer_norm_eps=1e-5, hidden_act='relu'): 
        
        super(EncoderBlock, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, attn_drop, proj_drop, qkv_bias)
        self.enc_ffn = PositionwiseFeedForward(d_model, d_ff, proj_drop, hidden_act)

        self.layernorm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
    
    def forward(self, enc_inputs, self_attn_mask=None): 
        norm_inputs = self.layernorm1(enc_inputs)#layer normalization is the first
        outputs = self.enc_self_attn(norm_inputs, norm_inputs, norm_inputs, self_attn_mask)
        tmp = enc_inputs + outputs
        
        norm_inputs = self.layernorm2(tmp)
        outputs = self.enc_ffn(norm_inputs)
        outputs = tmp + outputs
        return outputs



class TransformerDecoder(nn.Module): 
    """ Decoder module for Transformer. 
        This module can be used as an independent component. 
    """

    def __init__(self, layers_num=6, d_model=512, d_ff=2048, n_heads=8, d_k=64, d_v=64, 
                 attn_drop=0., proj_drop=0., qkv_bias=False, layer_norm_eps=1e-5, hidden_act='relu'): 
        """
            layers_num: indentical decoder layers
            d_model: dimension of each word
            d_ff: the dimension of the inner-layer in FFN
            n_heads: number of multi-heads
            d_k: dimension of key, which is same to the dimension of query
            d_v: dimension of value
            attn_drop: 0-1, 0 means without attention dropout
            proj_drop: 0-1, 0 means without linear dropout
            qkv_bias: whether linear for getting Q, K, V has bias
            layer_norm_eps: eps for nn.LayerNorm
            hidden_act: activation in FFN, use as 'relu', 'leakyrelu', 'gelu', 'quickgelu'
        """

        super(TransformerDecoder, self).__init__()
        self.decoder_list = nn.ModuleList([DecoderBlock(d_model, d_ff, n_heads, d_k, d_v, attn_drop, proj_drop, qkv_bias, layer_norm_eps, hidden_act) for _ in range(layers_num)])

    def forward(self, dec_inputs, enc_outputs, dec_self_mask=None, enc_dec_mask=None): 
        dec_outputs = dec_inputs
        for dec_layer in self.decoder_list: 
            dec_outputs = dec_layer(dec_outputs, enc_outputs, dec_self_mask, enc_dec_mask)
        
        return dec_outputs



class DecoderBlock(nn.Module): 
    """ Independent decoder block. 
        Contains self-attention, encoder-decoder attention and feed forward network. 
    """

    def __init__(self, d_model=512, d_ff=2048, n_heads=8, d_k=64, d_v=64, attn_drop=0., 
                 proj_drop=0., qkv_bias=False, layer_norm_eps=1e-5, hidden_act='relu'): 
        
        super(DecoderBlock, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, attn_drop, proj_drop, qkv_bias)#masked multi-head attention
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, attn_drop, proj_drop, qkv_bias)
        self.dec_ffn = PositionwiseFeedForward(d_model, d_ff, proj_drop, hidden_act)

        self.layernorm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layernorm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layernorm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
    
    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, enc_dec_mask=None): 
        norm_inputs = self.layernorm1(dec_inputs)
        tmp = dec_inputs + self.dec_self_attn(norm_inputs, norm_inputs, norm_inputs, self_attn_mask)#tmp: output of sublayer
        
        norm_inputs = self.layernorm2(tmp)
        memory = self.layernorm3(enc_outputs)#for we use pre-norm, the output of the encoder is not normalized
        tmp = tmp + self.enc_dec_attn(norm_inputs, memory, memory, enc_dec_mask)
        
        norm_inputs = self.layernorm4(tmp)
        outputs = tmp + self.dec_ffn(norm_inputs)
        return outputs



class PositionwiseFeedForward(nn.Module): 
    """ Implement of feed-forward network (FFN). 
        Function as one sub-layer. 

        Linear has bias. 
    """

    def __init__(self, d_model=512, d_ff=2048, dropout=0., hidden_act='relu'): 

        super(PositionwiseFeedForward, self).__init__()
        
        act = self.__get_activation(hidden_act)

        self.fc = nn.Sequential(nn.Linear(d_model, d_ff), 
                                act, 
                                nn.Linear(d_ff, d_model), 
                                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity())
    
    def forward(self, inputs): 
        outputs = self.fc(inputs)
        return outputs
    
    @staticmethod
    def __get_activation(hidden_act): 
        hidden_act = hidden_act.lower()
        if hidden_act == 'relu': 
            act = nn.ReLU()
        elif hidden_act == 'leakyrelu': 
            act = nn.LeakyReLU()
        elif hidden_act == 'gelu': #pytorch > 1.8.0
            act = nn.GELU()
        elif hidden_act == 'quickgelu': 
            act = QuickGELU()
        elif hidden_act == 'selu': 
            act = nn.SELU()
        else:
            raise AttributeError('Module "' + hidden_act + '" not found. See the source codes for more infomation. ')
        
        return act



class QuickGELU(nn.Module): 
    """ Trick from CLIP sources codes. 
        Very interesting. 
    """

    def __init__(self): 
        super(QuickGELU, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
