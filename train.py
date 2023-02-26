# -*- coding: utf-8 -*-
""" Training MAE. 
    This program can only run with DDP. 
"""

import os
import math
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision.transforms as transforms
from einops import rearrange

from dataset.ILSVRC2012Image import ILSVRC2012Image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import model as models_mae #see model.__init__.py
from eval.loss import MaskedMSELoss
from util.parser import TrainParser
from util.utils import distributed_init, Xavier_init, save_model, load_model
from util.param_groups import param_groups_wd

def main_worker(args): 
    """ One worker. 
    """
    #################### prepare device ####################
    distributed_init(args)

    print("{}".format(args).replace(', ', ',\n'))

    #terminal writer and tensorboard writer
    main_proc = args.rank == 0

    if main_proc: 
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    ############################################################

    
    #################### setup dataset and dataloader ####################
    train_transform = transforms.Compose([transforms.RandomResizedCrop(args.input_resolution, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    
    train_dataset = ILSVRC2012Image(root=args.data_path, 
                                    split='train', 
                                    resolution=args.input_resolution, 
                                    transform=train_transform)
    print('train dataset: ', train_dataset)

    #sampler and loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    print("train_sampler = %s" % str(train_sampler))

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               sampler=train_sampler, 
                                               num_workers=args.num_workers, 
                                               pin_memory=True, 
                                               drop_last=True)
    #equivalent batch size = batch_size * accumulation_steps * world_size
    ############################################################

    
    #################### setup model ####################
    model = models_mae.__dict__[args.model](input_resolution=args.input_resolution, 
                                            patch_size=args.patch_size, 
                                            channel=args.channel)
    
    #print model parameter
    summary(model, (args.channel, args.input_resolution, args.input_resolution))

    #deployment
    model = model.to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[args.local_rank], 
                                                      output_device=args.local_rank)
    unpacked_model = model.module
    ############################################################


    #################### setup optimizer and criterion ####################
    effective_batch_size = args.batch_size * args.accumulation_steps * args.world_size
    effective_lr = args.base_lr * effective_batch_size / 256
    print("effective batch size: %d" % effective_batch_size)
    print("effective learning rate: %.2e" % effective_lr)

    param_groups = param_groups_wd(model=unpacked_model, 
                                   weight_decay=args.weight_decay)
    
    optimizer = torch.optim.AdamW(param_groups, 
                                  lr=effective_lr, 
                                  betas=(0.9, 0.95))#learning rate will be changed by scheduler
    
    print(optimizer)

    #MSE loss with mask
    criterion = MaskedMSELoss()
    ############################################################


    #################### init or resume ####################
    if args.resume: 
        load_model(args=args, model=unpacked_model, optimizer=optimizer)
    elif args.pretrained: 
        load_model(args=args, model=unpacked_model, optimizer=None, strict=False)
    else: #initial with Xavier initialization
        Xavier_init(model=model)
    ############################################################


    #################### setup scheduler ####################
    """ Following the official implement, we also use a per iteration (instead of per epoch) lr scheduler. 
        Must be after loading optimizer checkpoint, cause args.last_epoch would be changed by load_model(). 
    """
    iter_length = math.ceil(len(train_loader) / args.accumulation_steps)#iter steps of one epoch
    lr_lambda = lambda step: step / iter_length / args.warmup_epoch if step / iter_length < args.warmup_epoch else 0.5 * (math.cos((step / iter_length - args.warmup_epoch) / (args.total_epoch - args.warmup_epoch) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=args.last_epoch * iter_length - 1)
    #note that last_epoch is set to -1 (args.last_epoch = 0) if train from scratch. 
    ############################################################
    

    #################### training ####################
    #loss, eff_batch_loss, total_loss
    print("Start training for %d epochs. " % args.total_epoch)
    start_time = datetime.now().replace(microsecond=0)

    torch.distributed.barrier()
    
    for epoch in range(args.last_epoch + 1, args.total_epoch + 1): 
        #train_one_epoch
        #shuffle for distributed training
        train_sampler.set_epoch(epoch)
        
        model.train()

        total_loss = 0
        eff_batch_loss = 0
        
        optimizer.zero_grad()
        for i, (imgs, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')) if main_proc else enumerate(train_loader): 
            #load data to device
            imgs = imgs.to(args.device, non_blocking=True)

            #run model
            preds, mask = model(imgs, mask_ratio=args.mask_ratio)

            #target
            tgts = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=args.patch_size, p2=args.patch_size)

            ##### norm_pix_loss #####
            """ use normalized pixel values can improve ImageNet accuracy by 1 percentage point! 
                0.5 for ViT-Large from paper. 
            """
            if args.norm_pix_loss: 
                tgts_mean = tgts.mean(dim=-1, keepdim=True)
                tgts_var = tgts.var(dim=-1, keepdim=True)
                tgts = (tgts - tgts_mean) / (tgts_var + 1.e-6) ** 0.5
            ###############

            loss = criterion(preds, tgts, mask)
            loss = loss / args.accumulation_steps
            loss_log = loss.detach().clone()
            loss.backward()

            eff_batch_loss += loss_log
            #accumulate
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader): 
                optimizer.step()
                optimizer.zero_grad()
                
                torch.distributed.all_reduce(eff_batch_loss, op=torch.distributed.ReduceOp.SUM)
                eff_batch_loss /= args.world_size
                total_loss += eff_batch_loss

                if main_proc: 
                    log_step = (i // args.accumulation_steps) + (epoch - 1) * iter_length
                    log_writer.add_scalar('loss', eff_batch_loss, log_step)
                
                eff_batch_loss.zero_()
                scheduler.step()#per iter
        
        #epoch end
        loss_mean = (total_loss * args.accumulation_steps / len(train_loader)).item()

        if main_proc: #use the first process as main process
            current_lr = scheduler.get_last_lr()[0]
            log_writer.add_scalar('lr', current_lr, epoch)

            #save txt
            log_stats = {'epoch': epoch, 'train_lr': current_lr, 'train_loss': loss_mean}
            
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f: 
                f.write(str(log_stats) + "\n")
            
            #only the master process saves the model
            if epoch % args.checkpoint_interval == 0: 
                save_model(args=args, epoch=epoch, model=unpacked_model, optimizer=optimizer)
        
        print('Epoch done %d / %d. Loss: %.6f. ' % (epoch, args.total_epoch, loss_mean))
    ############################################################

    if main_proc: 
        log_writer.close()
    
    total_time = datetime.now().replace(microsecond=0) - start_time
    print('Total training time {}'.format(total_time))



if __name__ == '__main__': 
    args = TrainParser().args
    assert args.model.startswith('mae')
    main_worker(args)