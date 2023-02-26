# -*- coding: utf-8 -*-
""" ImageNet 2012 dataset for our model. 
    Default transform is set in this dataset. Remove it for other use. 
"""

import os
import timeit
import torch
import torchvision.transforms as transforms
from dataset.CachedImageFolder import CachedImageFolder

#constants
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

class ILSVRC2012Image(CachedImageFolder): 
    """ Same as ImageNet from torchvision. 
        This dataset also has a static method to restore the image. 

        Folders: 
            ILSVRC2012_PATH 
            |-- train 
            |    |-- nxxxxxx
            |        |-- xxx.JPEG
            |        |-- .....
            |-- val
            |    |-- ......
            |
            |-- ILSVRC2012_devkit_t12.tar.gz
            |-- ......
     
    """
    def __init__(self, root, split='train', resolution=224, transform=None): 
        """ Only use train and val dataset. 
        """

        assert split == 'train' or split == 'val'#never use test

        default_transform = transforms.Compose([transforms.Resize([resolution, resolution]), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        
        print('Creating image folder... ')
        start_time = timeit.default_timer()
        super(ILSVRC2012Image, self).__init__(root=os.path.join(root, split), transform=transform if transform is not None else default_transform)
        end_time = timeit.default_timer()
        print('Iamge folder created! Building time %.2f seconds. ' % (end_time - start_time))
    
    def __getitem__(self, index): 
        img, cls = super(ILSVRC2012Image, self).__getitem__(index)#with transform
        return img, cls
    
    def __len__(self): 
        return super(ILSVRC2012Image, self).__len__()
    
    @staticmethod
    def denormalize(imgs): 
        """ imgs: (b, h, w, c)
        """
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=imgs.device)
        std = torch.tensor(IMAGENET_DEFAULT_STD, device=imgs.device)

        imgs = ((imgs * std + mean) * 255).int()
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        
        imgs = imgs.to(dtype=torch.uint8)
        return imgs