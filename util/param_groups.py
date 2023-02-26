# -*- coding: utf-8 -*-
""" implementation of layer-wise weight decay. 
    Learn from BEiT and MAE. 
"""

def param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75, no_weight_decay_list=[]): 
    """ Parameter groups for layer-wise lr decay, must be used with ViT model. 
        Note that the optimizer will not handle 'lr_scale' automatically. It should be handled in the training codes. 
        In the official implementation of MAE and BEiT, they use adjust_learning_rate() to set the lr in params. 
        But in our implementation, we reset the 'initial_lr' and use scheduler to adjust it. 
    """

    param_groups = {}

    """ For ViT, layer_id 0 is for patch_embed, cls_token and pos_embed. 
        1 ~ n for block layers. 
        n + 1 for mlp head and encoder_norm, in which the layer scale is 1. 
    """
    layer_scales = list(layer_decay ** i for i in range(len(model.blocks) + 2))
    layer_scales.reverse()

    for name, param in model.named_parameters(): 
        if not param.requires_grad: 
            continue

        if param.ndim == 1 or name in no_weight_decay_list: #LayerNorm has no weight decay. 
            g_decay = "no_decay"
            this_decay = 0.
        else: 
            g_decay = "decay"
            this_decay = weight_decay
        
        ####################
        """ From BEiT. 
            Most parameters names we use are the same as ViT in timm. 
            Note that the final 'head' in our implement is named 'mlp_head'. 
        """
        if name in ['cls_token', 'pos_embed'] or name.startswith('patch_embed'): 
            layer_id = 0
        elif name.startswith('blocks'): 
            layer_id = int(name.split('.')[1]) + 1
        else: 
            layer_id = len(model.blocks) + 1
        ####################

        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups: 
            this_scale = layer_scales[layer_id]

            param_groups[group_name] = {
                "lr_scale": this_scale, 
                "weight_decay": this_decay, 
                "params": [], 
            }
        
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())



def param_groups_wd(model, weight_decay=0.05, no_weight_decay_list=[]): 
    """ Parameter groups for weight decay. 
        LayerNorm has no weight decay. 
    """

    param_groups = {}

    for name, param in model.named_parameters(): 
        if not param.requires_grad: 
            continue

        if param.ndim == 1 or name in no_weight_decay_list: 
            group_name = "no_decay"
            this_decay = 0.
        else: 
            group_name = "decay"
            this_decay = weight_decay
        
        if group_name not in param_groups: 
            param_groups[group_name] = {
                "weight_decay": this_decay, 
                "params": [], 
            }
        
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())