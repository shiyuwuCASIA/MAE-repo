# -*- coding: utf-8 -*-
""" ImageFolder with cached annotations files. 
    Generate a cached list that prepared only once in advance and to be used for all runs. 

    From pytorch 1.9.0 torchvision files. 
    To get more detailed information, please see the raw defination of DatasetFolder and ImageFolder. 
"""

import os
import json
import fcntl
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS

class CachedDatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(CachedDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        #################### modified ####################
        #this code is stable for multi-processing, but a little slower
        cache_dataset_file = os.path.join(directory, 'instances.json')
        with open(cache_dataset_file, 'a+') as f: 
            fcntl.flock(f, fcntl.LOCK_EX)

            if os.path.getsize(cache_dataset_file) == 0: #no data, generate instances and save them

                instances = make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

                json.dump(instances, f)
                fcntl.flock(f, fcntl.LOCK_UN)
            else: #data exist, unlock and read the file
                fcntl.flock(f, fcntl.LOCK_UN)
                f.seek(0)
                instances = json.load(f)
        
        return instances
        ##################################################

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        #################### modified ####################
        cache_classes_file = os.path.join(dir, 'classes.json')
        with open(cache_classes_file, 'a+') as f: 
            fcntl.flock(f, fcntl.LOCK_EX)
            
            if os.path.getsize(cache_classes_file) == 0: 

                classes = [d.name for d in os.scandir(dir) if d.is_dir()]
                classes.sort()
                class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

                json.dump((classes, class_to_idx), f)
                fcntl.flock(f, fcntl.LOCK_UN)
            else: 
                fcntl.flock(f, fcntl.LOCK_UN)
                f.seek(0)
                classes, class_to_idx = json.load(f)
        
        return classes, class_to_idx
        ##################################################

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)



class CachedImageFolder(CachedDatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(CachedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples