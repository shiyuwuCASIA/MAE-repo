# -*- coding: utf-8 -*-
""" Finetuning ViT for classification. 
"""

import os
import math
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision.transforms as transforms
from timm.data.transforms_factory import transforms_imagenet_train
from timm.data.mixup import Mixup

from dataset.ILSVRC2012Image import ILSVRC2012Image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import model as models_vit #see model.__init__.py
from eval.loss import LabelSmoothingCrossEntropyLoss, SoftTargetCrossEntropyLoss
from eval.classify import ClassCalculator
from util.parser import FinetuningParser
from util.utils import distributed_init, Xavier_init, save_model, load_model
from util.param_groups import param_groups_lrd


def main_worker(args): 
    """ One worker. 
    """
    #################### prepare device ####################
    distributed_init(args)

    #terminal writer and tensorboard writer
    main_proc = args.rank == 0

    if main_proc: 
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    ############################################################

    
    #################### setup dataset and dataloader ####################
    #train dataset
    #get from timm
    train_transform = transforms_imagenet_train(img_size=args.input_resolution, 
                                                color_jitter=args.color_jitter, 
                                                auto_augment=args.auto_augment, 
                                                interpolation='bicubic', 
                                                mean=IMAGENET_DEFAULT_MEAN, 
                                                std=IMAGENET_DEFAULT_STD, 
                                                re_prob=args.re_prob, 
                                                re_mode=args.re_mode, 
                                                re_count=args.re_count)
    
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

    #val dataset
    val_transform = transforms.Compose([transforms.Resize(int(args.input_resolution * 256 / 224) if args.input_resolution <= 224 else args.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC), 
                                        transforms.CenterCrop(args.input_resolution), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    
    val_dataset = ILSVRC2012Image(root=args.data_path, 
                                  split='val', 
                                  resolution=args.input_resolution, 
                                  transform=val_transform)
    print('val dataset: ', val_dataset)

    #sampler and loader
    #dist_eval, see warning from MAE source codes
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    print("val_sampler = %s" % str(val_sampler))

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             sampler=val_sampler, 
                                             num_workers=args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=False)
    ############################################################

    
    #################### setup model ####################
    model = models_vit.__dict__[args.model](input_resolution=args.input_resolution, 
                                            patch_size=args.patch_size, 
                                            channel=args.channel, 
                                            drop_path=args.drop_path, 
                                            pos_embed=args.pos_embed, 
                                            feature_process=args.feature_process, 
                                            num_classes=args.num_classes)
    
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

    #layer-wise weight decay
    param_groups = param_groups_lrd(model=unpacked_model, 
                                    weight_decay=args.weight_decay, 
                                    layer_decay=args.layer_decay)
    
    optimizer = torch.optim.AdamW(param_groups, 
                                  lr=effective_lr, 
                                  betas=(0.9, 0.999))#learning rate will be changed by scheduler
    
    #reset 'lr' to effectuate layer decay
    #this 'lr' will be recorded as 'initial_lr' by scheduler, and actually 'lr' will be automatically changed when setting up scheduler
    for param_group in optimizer.param_groups: 
        if "lr_scale" in param_group: 
            param_group["lr"] *= param_group["lr_scale"]
    
    print(optimizer)

    #mix up
    if args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None: 
        print("Mixup is activated. ")
        mixup_fn = Mixup(mixup_alpha=args.mixup, 
                         cutmix_alpha=args.cutmix, 
                         cutmix_minmax=args.cutmix_minmax,
                         prob=args.mixup_prob, 
                         switch_prob=args.mixup_switch_prob, 
                         mode=args.mixup_mode, 
                         label_smoothing=args.smoothing, 
                         num_classes=args.num_classes)
    else: 
        mixup_fn = None
    
    #criterion
    if mixup_fn is not None: 
        criterion = SoftTargetCrossEntropyLoss()
    elif args.smoothing > 0: 
        criterion = LabelSmoothingCrossEntropyLoss(smoothing=args.smoothing)
    else: 
        criterion = torch.nn.CrossEntropyLoss()
    
    eval_criterion = torch.nn.CrossEntropyLoss()
    ############################################################


    #################### init or resume ####################
    if args.resume: 
        load_model(args=args, model=unpacked_model, optimizer=optimizer, strict=True)
    elif args.pretrained: 
        load_model(args=args, model=unpacked_model, optimizer=None, strict=False)
    else: #initial with Xavier initialization
        Xavier_init(model=model)
    ############################################################


    #################### setup scheduler ####################
    """ Following the official implement, we also use a per iteration (instead of per epoch) lr scheduler. 
        Must be placed after loading optimizer checkpoint, cause args.last_epoch would be changed by load_model(). 
    """
    iter_length = math.ceil(len(train_loader) / args.accumulation_steps)
    min_r = args.min_lr / effective_lr#min rate for scheduler
    lr_lambda = lambda step: step / iter_length / args.warmup_epoch if step / iter_length < args.warmup_epoch else min_r + (1 - min_r) * 0.5 * (math.cos((step / iter_length - args.warmup_epoch) / (args.total_epoch - args.warmup_epoch) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=args.last_epoch * iter_length - 1)
    #note that last_epoch is set to -1 (args.last_epoch = 0) if train from scratch. 
    ############################################################


    #################### training ####################
    #loss, eff_batch_loss, total_loss
    print("Start training for %d epochs. " % args.total_epoch)
    start_time = datetime.now().replace(microsecond=0)
    calculator = ClassCalculator(class_num=args.num_classes)

    torch.distributed.barrier()
    
    for epoch in range(args.last_epoch + 1, args.total_epoch + 1): 
        #train one epoch
        #shuffle for distributed training
        train_sampler.set_epoch(epoch)
        
        model.train()
        
        total_loss = 0
        eff_batch_loss = 0
        
        optimizer.zero_grad()
        for i, (imgs, cls) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')) if main_proc else enumerate(train_loader): 
            #load data to device
            imgs = imgs.to(args.device, non_blocking=True)
            cls = cls.to(args.device, non_blocking=True)

            #mix up
            if mixup_fn is not None: 
                imgs, cls = mixup_fn(imgs, cls)
            
            #run model, no mask
            pred = model(imgs)
            
            loss = criterion(pred, cls)
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


        ##### evaluation #####
        model.eval()

        val_total_loss = 0
        calculator.clear()
        with torch.no_grad(): 
            for _, (imgs, cls) in enumerate(tqdm(val_loader, desc=f'Evaluation {epoch}')) if main_proc else enumerate(val_loader): 
                imgs = imgs.to(args.device, non_blocking=True)
                cls = cls.to(args.device, non_blocking=True)

                #run model, no mask
                pred = model(imgs)

                loss = eval_criterion(pred, cls)
                val_total_loss += loss.detach().clone()

                calculator.fix(pred, cls)
        
        torch.distributed.all_reduce(val_total_loss, op=torch.distributed.ReduceOp.SUM)
        val_total_loss /= args.world_size
        calculator.all_reduce(args.device)
        
        val_loss_mean = (val_total_loss / len(val_loader)).item()
        acc1 = calculator.top_1_accuracy
        acc5 = calculator.top_5_accuracy
        ##### eval done #####

        if main_proc: #use the first process as main process
            current_lr = scheduler.get_last_lr()[0]
            log_writer.add_scalar('lr', current_lr, epoch)

            #save txt
            log_stats = {'epoch': epoch, 
                         'train_lr': current_lr, 
                         'train_loss': loss_mean, 
                         'val_loss': val_loss_mean, 
                         'acc1': acc1, 
                         'acc5': acc5}
            
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f: 
                f.write(str(log_stats) + "\n")
            
            #only the master process saves the model
            if epoch % args.checkpoint_interval == 0: 
                save_model(args=args, epoch=epoch, model=unpacked_model, optimizer=optimizer)
        
        print('Epoch done %d / %d. Train loss: %.6f, Val loss: %.6f, Top 1 accuracy: %.6f, Top 5 accuracy: %.6f. ' % (epoch, args.total_epoch, loss_mean, val_loss_mean, acc1, acc5))
    ############################################################

    if main_proc: 
        log_writer.close()
    
    total_time = datetime.now().replace(microsecond=0) - start_time
    print('Total training time {}'.format(total_time))



if __name__ == '__main__': 
    args = FinetuningParser().args
    assert args.model.startswith('vit')
    assert args.resume or args.pretrained
    main_worker(args)