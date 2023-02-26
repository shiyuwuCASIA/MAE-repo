# -*-coding: utf-8 -*-
""" Generate solid positional embedding. 
    Output data are tensors. 
"""

import torch
import torch.nn as nn
from einops import rearrange

class WordEmbedding(nn.Module): 
    """ A learnable input embedding. 
    """
    def __init__(self, vocab_size=1000, embed_dim=512): 
        super(WordEmbedding, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x): 
        x = self.embed(x) * self.embed_dim ** 0.5
        return x



class PatchEmbedding(nn.Module): 
    """ 2D Image to Patch Embedding. 
        Note that we do not use convolutional layer. 
    """

    def __init__(self, img_size=224, patch_size=16, input_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        assert img_size % patch_size == 0, 'Invalid image resolution %d with patch size %d. ' % (img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Linear(self.patch_size ** 2 * input_channels, embed_dim)

    def forward(self, x):
        #x: (B, C, H, W)
        img_patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        x = self.proj(img_patches)
        return x



class PositionalEncoding(): 
    """ Positional encoding for Transformer model. 
        All embeddings do not contain cls_token. 

        Current implementation：
            1d_absolute_sin_cos_embedding
            2d_absolute_sin_cos_embedding
            1d_learnable_embedding
    """

    @classmethod
    def get_1d_absolute_sin_cos_embedding(cls, pos=128, embed_dim=512): 
        """ Original Transformer encoding. 
            Generate positional encoding for origin Transformer. 
            PE(pos, 2i) = sin(pos / 10000^(2i / embed_dim))
            PE(pos, 2i + 1) = cos(pos / 10000^(2i / embed_dim))

            pos: a list of positions to be encoded, or an integer
        """

        #generate a list from an integer
        pos = torch.arange(0, pos) if isinstance(pos, int) else pos

        pos_emb = torch.zeros((pos.size(0), embed_dim))

        position = pos.unsqueeze(1)#(max_len, 1)

        #make sure embed_dim is an even number
        div_term = torch.pow(1 / 10000, torch.arange(0, embed_dim, 2) / embed_dim)#embed_dim / 2
        pos_emb[:, 0::2] = torch.sin(position * div_term)#(max_len, embed_dim / 2)
        pos_emb[:, 1::2] = torch.cos(position * div_term)#(max_len, embed_dim / 2)
        
        pos_emb = pos_emb.unsqueeze(0)#(1, max_len, embed_dim), the first channel is used for batch
        #return data is on CPU
        return pos_emb
    
    @classmethod
    def get_2d_absolute_sin_cos_embedding(cls, h, w, embed_dim, flatten=True): 
        """ Embedding used by MAE. 

            If flatten is True, the output is 1d. (always true)
        """

        #make sure embed_dim can be divided by 4
        assert embed_dim % 4 == 0, 'embed_dim must be a multiple of 4. '

        pos_emb = torch.zeros((h * w, embed_dim))
        m1, m2 = torch.meshgrid(torch.arange(h), torch.arange(w))

        h_emb = cls.get_1d_absolute_sin_cos_embedding(m1.reshape(-1), embed_dim // 2)
        w_emb = cls.get_1d_absolute_sin_cos_embedding(m2.reshape(-1), embed_dim // 2)

        pos_emb[:, :embed_dim // 2] = h_emb
        pos_emb[:, embed_dim // 2:] = w_emb

        #restore to original shape
        if not flatten: 
            pos_emb = pos_emb.reshape(h, w, embed_dim)
        
        pos_emb = pos_emb.unsqueeze(0)

        return pos_emb
    
    @classmethod
    def get_1d_learnable_embedding(cls, pos_len, embed_dim): 
        """ A learnable embedding. 
            No embdding for cls token. 
        """

        pos_emb = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        
        return pos_emb