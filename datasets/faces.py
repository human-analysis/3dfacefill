import numpy as np
import os, glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from datasets import loaders
import utils
from utils import writetextfile, readtextfile
import time
import random
random.seed()

__all__ = ['Faces']

class Faces(Dataset):
    """docstring for _3DMM"""
    def __init__(self, args, root, transform, split='train'):
        super(Faces, self).__init__()
        self.root = root
        self.transform = transform
        self.totensor = torchvision.transforms.ToTensor()

        image_list = []
        for _, dirs, _ in os.walk(self.root):
            for dir in dirs:
                files = glob.glob(os.path.join(self.root, dir, '*'))
                image_list.append(files)
            break
        self.image_filename = np.sort(np.concatenate(image_list, axis=0))

        self.train_percent = args.train_dev_percent
        self.split = split
        if split == 'train':
            self.offset = 0
            self.len = int(self.image_filename.shape[0] * self.train_percent)
        elif split == 'test':
            self.offset = int(self.image_filename.shape[0] * self.train_percent)
            self.len = int(self.image_filename.shape[0] * (1 - self.train_percent))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx += self.offset
        image = loaders.loader_image(self.image_filename[idx])
        if self.transform:
            if hasattr(self.transform, 'get_class_fullname') and 'albumentations' in self.transform.get_class_fullname():
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)

        return image, self.image_filename[idx]
