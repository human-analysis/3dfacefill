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
from datasets import CelebAMask, Faces

__all__ = ['CelebACombined']

class CelebACombined(Dataset):
    """docstring for _3DMM"""
    def __init__(self, args, root, transform, split='train'):
        super(CelebACombined, self).__init__()
        self.celeba_root = '/research/hal-datastage/datasets/processed/CelebA/celebahq_crop/all_images/'
        self.celebahq_root = '/research/hal-datastage/datasets/original/CelebAMaskHQ/'
        self.transform = transform

        self.celebahq = CelebAMask(args, self.celebahq_root, transform, split)
        self.celeba = Faces(args, self.celeba_root, transform, split)

        # self.__getitem__(100)

    def __len__(self):
        return self.celeba.__len__() + self.celebahq.__len__()

    def __getitem__(self, idx):
        if idx > self.celeba.__len__():
            idx -= self.celeba.__len__()
            image, _, _, _, shape, shape_conf, filename = self.celebahq.__getitem__(idx)
        else:
            image, shape, shape_conf, filename = self.celeba.__getitem__(idx)

        return image, shape, shape_conf, filename
