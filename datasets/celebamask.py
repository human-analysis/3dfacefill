from glob import glob
from collections import defaultdict
from sklearn.utils import shuffle
import numpy as np
import os, glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import loaders
import pickle
import math
from torchvision.transforms import ToTensor, Normalize

__all__ = ['CelebAMask']

class CelebAMask(Dataset):
    """docstring for CelebAMask"""
    def __init__(self, args, root, transform, split='train'):
        super(CelebAMask, self).__init__()
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(self.root, "CelebA-HQ-img")
        self.shape_dir = os.path.join(self.root, "CelebA-HQ-shapes")
        self.finetuned_gt = os.path.join(self.root, "finetuned_gt")
        self.pre_anno_dir = os.path.join(self.root, "CelebAMask-HQ-mask-anno") #"occl_mask"
        self.post_anno_dir = os.path.join(self.root, "occlusions")
        self.pre_anno_occ = ['hat', 'cloth', 'eye_g', 'mouth', 'hair', 'neck', 'ear_r', 'neck_l']
        self.occ_list = ['eye_g']
        self.post_anno_occ = ['facial_hair', 'hand', 'mic', 'cloth', 'phone']#, 'misc', 'unsure']
        self.totensor = ToTensor()
        self.normalize = Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

        # processed_file_path = os.path.join(self.post_anno_dir, 'processed.txt')
        # if os.path.exists(processed_file_path):
        #     self.num_processed_file = open(processed_file_path, 'r+')
        #     lines = self.num_processed_file.read().splitlines()
        #     self.len = int(lines[-1])
        self.train_percent = args.train_dev_percent
        self.split = split
        if split == 'train':
            self.offset = 0
            self.len = int(30000 * self.train_percent)
        elif split == 'test':
            self.offset = int(30000 * self.train_percent)
            self.len = int(30000 * (1 - self.train_percent))

        # self.__getitem__(100)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx += self.offset
        image_path = os.path.join(self.image_dir, '{}.jpg'.format(idx))
        shape_path = os.path.join(self.shape_dir, '{}.npy'.format(idx))
        occl_sub_dir = math.floor(idx / 2000)
        pre_anno_dir = os.path.join(self.pre_anno_dir, '{}/{:05d}'.format(occl_sub_dir, idx))
        post_anno_dir = os.path.join(self.post_anno_dir, '{}/{:05d}'.format(occl_sub_dir, idx))
        image = loaders.loader_image(image_path)
        params_dir = os.path.join(self.finetuned_gt, '{}'.format(idx))

        skin_path = '{}_skin.png'.format(pre_anno_dir)
        skin = np.array(loaders.loader_image(skin_path))[:,:,0]
        occ = np.zeros_like(skin)

        for att in self.pre_anno_occ:
            att_path = '{}_{}.png'.format(pre_anno_dir, att)
            if os.path.exists(att_path):
                att = np.array(loaders.loader_image(att_path))[:,:,0]
                # skin[att == 255] = 0

        # for att in self.occ_list:
        #     att_path = '{}_{}.png'.format(pre_anno_dir, att)
        #     if os.path.exists(att_path):
        #         att = np.array(loaders.loader_image(att_path))[:,:,0]
        #         occ[att == 255] = 255

        for att in self.post_anno_occ:
            att_path = '{}_{}.png'.format(post_anno_dir, att)
            if os.path.exists(att_path):
                att = np.array(loaders.loader_image(att_path))[:,:,0]
                # skin[att == 255] = 0
                occ[att == 255] = 255

        # with open(os.path.join(params_dir, 'params.pkl'), 'rb') as param_file:
        #     params = pickle.load(param_file)
        #     shape = params['shape']
        #     rotation = params['rotation']
        #     scale = params['scale']
        #     translation = params['translation']
        #     il = params['il']
        # texture = loaders.loader_image(os.path.join(params_dir, 'texture.jpg'))
        # texture = self.normalize(self.totensor(texture))

        mask = skin[:,:,np.newaxis].repeat(3, axis=2)
        occ = occ[:,:,np.newaxis].repeat(3, axis=2)
        if self.transform:
            augmented = self.transform(image=np.array(image), img_mask=mask, occ_mask=occ)
            image = augmented['image']
            mask = 0.5*(augmented['img_mask']+1)
            occ = 0.5*(augmented['occ_mask']+1)

        shape_params = np.load(shape_path, allow_pickle=True).item()

        # return image, mask, image, mask, shape, scale, rotation, translation, il, texture, image_path
        return image, mask, occ, image, mask, shape_params['vertex'], shape_params['conf'], image_path
