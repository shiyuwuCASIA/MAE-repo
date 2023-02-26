# -*- coding: utf-8 -*-
""" Parser used by this program. 
"""

import argparse

class ModelParser():
    """ Setup parser. 
    """

    def __init__(self): 
        self.parser = argparse.ArgumentParser()

        """ base options """
        #environment
        self.parser.add_argument('--model', type=str, default='mae_vit_base_patch16', help='Name of model to train. ')
        #directory
        self.parser.add_argument('--output_dir', type=str, default='./output_dir', help='The directory for saving checkpoints. ')
        self.parser.add_argument('--log_dir', type=str, default='./runs', help='The directory for saving tensorboard logs. ')
        #load pretrained model
        self.parser.add_argument('--pretrained', type=self._str2bool, default=False, help='Use pretrained model. ')
        self.parser.add_argument('--resume', type=self._str2bool, default=False, help='Resume from checkpoint. ')
        self.parser.add_argument('--load_file', type=str, default='', help='File for pretrained or resume model. ')
        #print info
        self.parser.add_argument('--disable_branch_print', type=self._str2bool, default=True, help='Disable print from branch processes in the terminal. ')


        """ DDP options """
        #no need to set these arguments, cause they will be initialized automatically
        #see util.utils.distributed_init() for more information
        self.parser.add_argument('--world_size', type=int, default=1, help='World size for multiprocessing. ')
        self.parser.add_argument('--master_addr', type=str, default='env://', help='Master address for multiprocessing. ')
        self.parser.add_argument('--master_port', type=str, default='23456', help='Master port for multiprocessing. ')
        self.parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed data parallel. ')
        self.parser.add_argument('--rank', type=int, default=0, help='Rank for distributed data parallel. ')
        #device will be added at distributed_init()
        

        """ model options & dataset options """
        self.parser.add_argument('--data_path', type=str, default='./data', help='Path of the dataset. For ImageNet only, do not add "/train", "/val" or "/test". ')
        self.parser.add_argument('--input_resolution', type=int, default=224, help='Input image size. ')
        self.parser.add_argument('--patch_size', type=int, default=16, help='Size of each patch. ')
        self.parser.add_argument('--channel', type=int, default=3, help='Input image channel. ')
        #other options can be changed when you define the model


        """ optimizer options """
        self.parser.add_argument('--base_lr', type=float, default=1.5e-4, help='Base learning rate for training. ')
        #we do not use min_lr
        self.parser.add_argument('--warmup_epoch', type=int, default=5, help='Epoches for warming up. ')
        #no need to care about when use resume
        self.parser.add_argument('--last_epoch', type=int, default=0, help='Last epoch for resuming training. Set 0 for training from scratch. ')#optimizer last_epoch = args.last_epoch - 1
        

        """ dataloader options """
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size for one GPU(device). Effective batch size is batch_size * accumulation_steps * GPU_num (or world_size). ')
        self.parser.add_argument('--num_workers', type=int, default=4,  help='Number of workers in dataloader. ')
        #other options like data_path and device should be specified in the dataset file


        """ training options """
        self.parser.add_argument('--total_epoch', type=int, default=400, help='Total epoches. ')
        self.parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints). ')
        self.parser.add_argument('--checkpoint_interval', type=int, default=20, help='Interval between saving model weights. ')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed for the main processes. ')
    
    @property
    def args(self):
        return self.parser.parse_args()
    
    @staticmethod
    def _str2bool(value): 
        if value.lower() in ('yes', 'true', 't', 'y', '1'): 
            return True
        elif value.lower() in ('no', 'false', 't', 'y', '0'): 
            return False
        else: 
            raise argparse.ArgumentTypeError('Unsupported value encountered. ')



class TrainParser(ModelParser): 
    def __init__(self): 
        super(TrainParser, self).__init__()
        #model
        self.parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masked ratio for image. ')
        self.parser.add_argument('--norm_pix_loss', type=self._str2bool, default=True, help='Use per-patch normalized pixels as targets for computing loss. ')

        #training
        self.parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay. ')



class FinetuningParser(ModelParser): 
    def __init__(self): 
        super(FinetuningParser, self).__init__()
        #model
        self.parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate. ')
        self.parser.add_argument('--feature_process', type=str, default='mean', help='Feature process for ViT. ("cls" or "mean")')
        self.parser.add_argument('--pos_embed', type=str, default='learnable', help='Type of positional embedding for ViT. ("learnable", "1d", "2d" or "none")')
        self.parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes for output. (default: 1000)')


        """ From MAE source codes. 
        """
        #training
        self.parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay. ')
        self.parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer-wise lr decay. ')
        self.parser.add_argument('--min_lr', type=float, default=1e-6, help='Lower lr bound for cyclic schedulers that hit 0. ')#important for finetuning

        # augmentation
        self.parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (enabled only when not using Auto/RandAug)')
        self.parser.add_argument('--auto_augment', type=str, default='rand-m9-mstd0.5-inc1', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
        self.parser.add_argument('--re_prob', type=float, default=0.25, help='Random erase prob (default: 0.25)')
        self.parser.add_argument('--re_mode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
        self.parser.add_argument('--re_count', type=int, default=1, help='Random erase count (default: 1)')

        #mixup & cutmix
        self.parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
        self.parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
        self.parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
        self.parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
        self.parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
        self.parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
        self.parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
