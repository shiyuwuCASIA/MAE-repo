# -*-coding: utf-8 -*-
""" Functions for devices preparing, model initializing, saving and loading. 
"""

import os
import random
import collections
import builtins
from datetime import datetime
from pytz import timezone
import numpy as np
import torch
import torch.nn as nn

#for environment
def distributed_init(args): 
    """ Prepare device for training or testing using DDP, while each process only runs on one GPU. 
        
        GPU_ids should be continuous, like 0, 1, 2, ...
        Use os.environ['CUDA_VISIBLE_DEVICES'] to make it specific. 
    """

    #get args from environment
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.master_addr = os.environ['MASTER_ADDR']#could set by rzdv_endpoint
    args.master_port = os.environ['MASTER_PORT']

    #init
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    if args.local_rank == 0: 
        print('Initial process group with tcp://{}:{}. '.format(args.master_addr, args.master_port))
    
    #torchrun can automatically fill the parameters
    torch.distributed.init_process_group(backend='nccl', 
                                         init_method='tcp://{}:{}'.format(args.master_addr, args.master_port), 
                                         world_size=args.world_size, 
                                         rank=args.rank)
    
    is_master = args.rank == 0
    set_print(not args.disable_branch_print or is_master)

    #set seed
    seed = args.seed + args.rank
    set_seed(seed)



def set_seed(seed): 
    """ Fix the seed for reproducibility. 
    """

    #python & numpy
    random.seed(seed)
    np.random.seed(seed)

    #pytorch
    """ Use torch.manual_seed() to seed the RNG for all device (both CPU and CUDA). 
    """
    torch.manual_seed(seed)

    """ Our model does not contain convolutional layer (even at the patch embedding module). 
        No need to set torch.backends.cudnn. 
    """
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False



def set_print(is_master): 
    """ This function disables printing when not in master process. 
        This will not affect tqdm print. 
        Learn from MAE. 
    """

    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force: 
            beijing = timezone('Asia/Shanghai')#we live here, welcome
            now = datetime.utcnow().astimezone(beijing).strftime("%H:%M:%S")
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print



#for model
def save_model(args, epoch, model, optimizer=None): 
    """ Save the model and the optimizer. 
        Always save model without "module". 

        Note that all attributes and functions on the outmost layer of the model should not be named 'module'! 
        Must implement "output_dir", "model" in option "args". 
    """

    os.makedirs(args.output_dir, exist_ok=True)
    
    #filename and path
    save_filename = '%s_checkpoint_%s.pth' % (args.model, epoch)
    save_path = os.path.join(args.output_dir, save_filename)
    
    #save data
    data_to_save = {'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                    'optimizer': optimizer.state_dict() if optimizer is not None else collections.OrderedDict(), 
                    'epoch': epoch, 
                    'args': args}
    
    torch.save(data_to_save, save_path)
    print('Save checkpoint: ' + save_filename + ' done. ')



def load_model(args, model, optimizer=None, strict=True): 
    """ Load the parameters of the model and optimizer. 
        
        To avoid mistake, always loading model on cuda:0 or cpu is a good habit. 
        Note that all attributes and functions on the outmost layer of the model should not be named 'module' ! 
        Must implement "load_file" for resuming. 
    """
    
    checkpoint = torch.load(args.load_file, map_location='cpu')#always cpu is well

    ########################################
    """ If you have good habit, codes below are useless. 
        That means you always save or load the model without warpping it. 
    """

    #load model
    state_dict = checkpoint['model']
    key_list = list(state_dict.keys())
    assert len(key_list) > 0, 'No key in state dictionary. '#must has at least one key-value pair
    key = key_list[0]#use the first key to check module

    
    if hasattr(model, 'module') and key.split('.')[0] != 'module': 
        print('Add "module." to state dict. ')
        new_state_dict = collections.OrderedDict()
        #add 'module' to each key
        for k, v in state_dict.items(): 
            new_k = 'module.' + k
            new_state_dict[new_k] = v
        state_dict = new_state_dict
    elif not hasattr(model, 'module') and key.split('.')[0] == 'module': 
        print('Remove "module." from state dict. ')
        new_state_dict = collections.OrderedDict()
        #remove 'module' from each key
        for k, v in state_dict.items(): 
            new_k = k[7:]
            new_state_dict[new_k] = v
        state_dict = new_state_dict
    #else: do nothing
    ########################################

    print('Resume checkpoint from "%s". ' % args.load_file)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    #infomation
    if strict == False: 
        print('Unexpected keys: ', unexpected_keys)
        print('Missing keys: ', missing_keys)

    #optimizer
    if args.resume: 
        if optimizer is not None: 
            print('Resume optimizer from "%s". ' % args.load_file)
            optim_state_dict = checkpoint['optimizer']
            optimizer.load_state_dict(optim_state_dict)
        else: 
            print('No optimizer to resume. ')
        
        args.last_epoch = checkpoint['epoch']
        


def Xavier_init(model): 
    """ Use Xavier initialization. 
    """

    for m in model.modules(): 
        if isinstance(m, nn.Linear): 
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): 
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)