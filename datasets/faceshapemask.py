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
from datasets import CelebAMask

__all__ = ['FaceShapeMask']

class FaceShapeMask(Dataset):
    """docstring for _3DMM"""
    def __init__(self, args, root, transform, split='train'):
        super(FaceShapeMask, self).__init__()
        self.root = root
        self.transform = transform
        self._3dmm_root = '/research/hal-datastage/datasets/processed/3DMM'
        self.facemask = CelebAMask(args, root, transform, split)
        # self.facemask = CelebAMaskAug(args, root, transform, split=split)
        # self.facemask = CelebAMask3DAug(args, root, transform, split=split)

        # shape and exp
        dataset = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']
        dataset_num = len(dataset)
        shape = [0] * dataset_num
        exp = [0] * dataset_num
        for i in range(dataset_num):
            shape[i], exp[i] = self.load_300W_LP_dataset(dataset[i])
        self.all_shape_para = np.concatenate(shape, axis=0)
        self.all_exp_para = np.concatenate(exp, axis=0)

        # self.__getitem__(100)

    def __len__(self):
        return self.facemask.__len__()

    def __getitem__(self, idx):
        # image, mask, orig_image, orig_mask, orig_face, shape, scale, rotation, translation, il, texture, filename = self.facemask.__getitem__(idx)   #, shape, scale, pose, il, texture
        # image, mask, orig_image, orig_mask, shape, scale, rotation, translation, il, texture, filename = self.facemask.__getitem__(idx)
        image, mask, orig_image, orig_mask, shape, shape_conf, filename = self.facemask.__getitem__(idx)

        _3dmm_idx = idx % self.all_shape_para.shape[0]

        # return image, mask, orig_image, orig_mask, shape, scale, rotation, translation, il, texture, self.all_shape_para[_3dmm_idx], self.all_exp_para[_3dmm_idx], filename
        return image, mask, orig_image, orig_mask, shape, shape_conf, filename

    def load_300W_LP_dataset(self, dataset):
        # print('Loading ' + dataset + '...')
        fd = open(os.path.join(self._3dmm_root, 'filelist', '{}_param.dat'.format(dataset)))
        all_paras = np.fromfile(file=fd, dtype=np.float32)
        fd.close()

        idDim = 1
        mDim  = idDim + 8
        poseDim = mDim + 7
        shapeDim = poseDim + 199
        expDim = shapeDim + 29
        texDim = expDim + 40
        ilDim  = texDim + 10

        all_paras = all_paras.reshape((-1,ilDim)).astype(np.float32)
        if 'HELEN' in dataset:
            all_paras = np.delete(all_paras, np.arange(37295, 37312), 0)
            all_paras = np.delete(all_paras, np.arange(29703, 29721), 0)
            all_paras = np.delete(all_paras, np.arange(17405, 17423), 0)
            all_paras = np.delete(all_paras, np.arange(7813, 7831), 0)
            all_paras = np.delete(all_paras, np.arange(5688, 5701), 0)
        elif 'IBUG' in dataset:
            all_paras = np.delete(all_paras, np.arange(1059, 1077), 0)
            all_paras = np.delete(all_paras, np.arange(1045, 1059), 0)
            all_paras = np.delete(all_paras, np.arange(433, 448), 0)

        pid = all_paras[:,0:idDim]
        m = all_paras[:,idDim:mDim]
        pose = all_paras[:,mDim:poseDim]
        shape = all_paras[:,poseDim:shapeDim]
        exp = all_paras[:,shapeDim:expDim]
        tex = all_paras[:,expDim:texDim]
        il  = all_paras[:,texDim:ilDim]

        return shape, exp
