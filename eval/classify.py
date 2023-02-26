# -*- coding: utf-8 -*-
""" Calculator for classification. 
    The input of the calculator could be on GPU, but the attributes are on CPU. 
"""

import torch

class ClassCalculator(): 
    def __init__(self, class_num=1000, softmax=True): 
        self.class_num = class_num
        self.softmax = softmax
        self.top_1_num = 0
        self.top_5_num = 0
        self.total = 0
    
    def fix(self, pred, label): 
        """ Calculate the predicted data. Using reshape to omit batch dimension. 
            
            pred: tensor (length, class_num)
            label: one dim tensor
        """

        assert pred.size(0) == label.size(0) and pred.size(1) == self.class_num, 'Input is invalid. '
        pred = pred.softmax(dim=-1) if self.softmax else pred

        _, top_1_cls = pred.max(dim=-1)
        self.top_1_num += (top_1_cls == label).sum().item()

        if self.class_num >= 5: 
            _, top_5_cls = pred.topk(k=5, dim=1)
            self.top_5_num += (top_5_cls == label.unsqueeze(-1)).sum().item()
        
        self.total += len(pred)
        #for batch view if you want
        return top_1_cls == label
    
    @property
    def top_1_accuracy(self): 
        return self.top_1_num / (self.total + 1e-6)
    
    @property
    def top_5_accuracy(self): 
        return self.top_5_num / (self.total + 1e-6) if self.class_num >= 5 else None
    
    def clear(self): 
        self.top_1_num = 0
        self.top_5_num = 0
        self.total = 0
    
    def all_reduce(self, device): 
        """ Only for DDP evaluation. 
        """

        x_reduce = torch.tensor(self.top_1_num, device=device)
        torch.distributed.all_reduce(x_reduce, op=torch.distributed.ReduceOp.SUM)
        self.top_1_num = x_reduce.item()

        x_reduce = torch.tensor(self.top_5_num, device=device)
        torch.distributed.all_reduce(x_reduce, op=torch.distributed.ReduceOp.SUM)
        self.top_5_num = x_reduce.item()

        x_reduce = torch.tensor(self.total, device=device)
        torch.distributed.all_reduce(x_reduce, op=torch.distributed.ReduceOp.SUM)
        self.total = x_reduce.item()