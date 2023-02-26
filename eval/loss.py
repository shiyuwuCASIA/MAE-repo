# -*- coding: utf-8 -*-
""" Loss function for user with pytorch <=1.9.0. 
    All data here are torch.tensor or torch.cuda.tensor. 

    Pytorch 1.10 has supported label smoothing and soft target in torch.nn.CrossEntropyLoss. 
    So there is no need to use these functions here. 
"""

import torch
import torch.nn as nn

class LabelSmoothingCrossEntropyLoss(nn.Module): 
    """ Calulate cross-entropy loss with label smoothing. 
        If smoothing=0, it is same to the standard CrossEntropyLoss. 
    """

    def __init__(self, ignore_index=-100, smoothing=0.0, reduction='mean'): 
        """ reduction: 'mean', 'sum', 'none'
        """

        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert reduction == 'mean' or reduction == 'sum' or reduction == 'none', 'Invalid reduction: ' + reduction

        self.ignore_index = ignore_index
        self.eps = smoothing
        self.reduction = reduction

        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, target): 
        """ Input data x is a 2-dim tensor, in which batch and max_len are fused to one dimension. 
            Target is a one-dim tensor for classes. 
        """
        
        assert x.size(-1) > 1, 'This function is not for one-class prediction. '
        
        #log softmax
        log_distribution = -self.log_softmax(x)

        #probability of each prediction
        label = nn.functional.one_hot(target, x.size(-1)).float()
        label = torch.clamp(label, min=self.eps / (x.size(-1) - 1), max=1 - self.eps)

        loss = (log_distribution * label).sum(-1)

        #reduction
        if self.reduction == 'mean': 
            loss = loss[target != self.ignore_index].mean()
        elif self.reduction == 'sum': 
            loss = loss[target != self.ignore_index].sum()
        else: #reduction = 'none'
            loss[target == self.ignore_index] = 0
        
        return loss



class SoftTargetCrossEntropyLoss(nn.Module): 
    """ Soft target cross entropy loss. 
    """

    def __init__(self, reduction='mean'): 
        """ reduction: 'mean', 'sum', 'none'
        """

        super(SoftTargetCrossEntropyLoss, self).__init__()
        assert reduction == 'mean' or reduction == 'sum' or reduction == 'none', 'Invalid reduction: ' + reduction
        self.reduction = reduction

        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, target): 
        """ Input data x is a 2-dim tensor, in which batch and max_len are fused to one dimension. 
            The size of the target is same as input x. 
        """
        
        assert x.size(-1) > 1, 'This function is not for one-class prediction. '

        #log softmax
        log_distribution = -self.log_softmax(x)

        loss = (log_distribution * target).sum(-1)

        #reduction
        if self.reduction == 'mean': 
            loss = loss.mean()
        elif self.reduction == 'sum': 
            loss = loss.sum()
        #else: 'none'
        
        return loss



class MaskedMSELoss(nn.Module): 
    """ MSE loss with masked pixels. 

        Within the mask, use 0 for dropout and 1 for reservation. 
        If mask is None, same as MSELoss. 
    """

    def __init__(self, reduction='mean'): 
        """ reduction: 'mean', 'sum', 'none'
        """

        super(MaskedMSELoss, self).__init__()
        assert reduction == 'mean' or reduction == 'sum' or reduction == 'none', 'Invalid reduction: ' + reduction
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, x, target, mask=None): 
        if mask == None: 
            mask = torch.ones(x.size(), device=x.device)
        
        assert x.size() == mask.size()
        assert mask.sum() > 0
        
        loss = self.mse_loss(x, target)
        loss = loss * mask.float()

        #reduction
        if self.reduction == 'mean': 
            loss = loss.sum() / mask.sum()
        elif self.reduction == 'sum': 
            loss = loss.sum()
        else: #reduction = 'none'
            loss[mask == 0] = 0
        
        return loss