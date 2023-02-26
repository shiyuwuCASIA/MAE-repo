# Unofficial PyTorch implementation of [Masked Autoencoders](https://arxiv.org/abs/2111.06377)

This is an unofficial PyTorch/GPU implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). We made it to thoroughly understand the training strategy of MAE. As the author is a new comer to **Transformer & ViT**, models are built from scratch in this implementation, which might be useful for other beginners. 

We learned a lot from the [official repository](https://github.com/facebookresearch/mae) and the [unofficial repository](https://github.com/pengzhiliang/MAE-pytorch) of MAE, thanks very much! The repository [timm](https://github.com/rwightman/pytorch-image-models) is also a treasure-house. We use the data augmentation and mixup from it when finetuning. 

We also made some further studys on MIM, but failed. (Laugh)

## Note

* This implementation is based on PyTorch >= 1.10.0. It could also be used with a lower version if you fix the AdamW bugs (fixed at 1.9.0) and abandon GELU (implement at 1.8.0). 

* We use `torchrun` as our DDP starter. `torch.distributed.launch` is also OK for old version. 

* This implementation needs more consumption than works above cause we do not use mixed-precision-training. 

## Run

### Setup

```
pip install -r requirements.txt
```

### Pretrain

To pre-train ViT-base with DDP, run the following on 1 node with 8 GPUs. 
```
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --batch_size=128 \
    --accumulation_steps=4 \
    --num_workers=8 \
    --total_epoch=400 \
    --checkpoint_interval=10 \
    --warmup_epoch=40 \
    --data_path=${data_path} \
    --output_dir=${output_dir} \
    --log_dir=${log_dir}
```

If on 2 nodes with 8 GPUs each, one node runs: 
```
OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --rzdv_endpoint='${master_addr}:${master_port}' train.py \
    --batch_size=128 \
    --accumulation_steps=2 \
    --num_workers=8 \
    --total_epoch=400 \
    --checkpoint_interval=10 \
    --warmup_epoch=40 \
    --data_path=${data_path} \
    --output_dir=${output_dir} \
    --log_dir=${log_dir}
```

The other node runs: 
```
OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --rzdv_endpoint='${master_addr}:${master_port}' train.py \
    ...... \
```

We also implement `--resume` and one can easily resume the training from unexpected breakpoint. See `train.py` or `finetuning.py` for more information. 

### Finetuning
Same as pretraining, run with
```
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 finetuning.py \
    --model=vit_base_patch16
    --batch_size=64 \
    --accumulation_steps=2 \
    --num_workers=8 \
    --total_epoch=100 \
    --checkpoint_interval=1 \
    --warmup_epoch=5 \
    --base_lr=1e-3 \
    --min_lr=1e-6 \
    --data_path=${data_path} \
    --output_dir=${output_dir} \
    --log_dir=${log_dir} \
    --pretrained=True \
    --load_file=${checkpoint.pth} \
    --layer_decay=0.75 \
    --weight_decay=0.05 \
    --drop_path=0.1 \
    --re_prob=0.25  \
    --mixup=0.8 \
    --cutmix=1.0
```
Time to finetune Vit-base model for 100 epoches on 8 A100 is 11h36m (including evaluation of every epoch). 

## Result
ViT-Large could be coming soon. No power to update. 
|   model  | pretrain (epoch/warmup) | finetune (epoch/warmup) | top-1 accuracy | top-5 accuracy| pretrain weight | finetune weight |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| vit-base |   400/40   |   100/5   |   83.15%  |   96.50%  | [BaiduYun](https://pan.baidu.com/s/1PbqyaMU2wHCRmAo4O4AU9A) (code: mae9) | [BaiduYun](https://pan.baidu.com/s/1-6It7w5QZcC30nvypgywkQ) (code: mae9)|
| vit-large | 400/40 | 50/5 | - | - | Todo | Todo |
