# checkpoints.py

import os
import torch
from collections import OrderedDict


class Checkpoints:
    def __init__(self, args):
        self.dir_save = args.save_dir
        self.ckpt_path = args.resume
        self.save_results = args.save_results
        self.cuda = args.cuda
        self.device = args.device

        if self.save_results and not os.path.isdir(self.dir_save) and args.rank == 0:
            os.makedirs(self.dir_save)

    def save_ckpt(self, epoch, ckpt_dict, prefix='checkpoint', add_epoch=True):
        if add_epoch:
            torch.save(ckpt_dict, '%s/%s_%d.pth' % (self.dir_save, prefix, epoch))
        else:
            torch.save(ckpt_dict, '%s/%s.pth' % (self.dir_save, prefix))

    def load_ckpt(self, filename=None):
        if filename is None:
            filename = self.ckpt_path
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=self.device)

            return checkpoint
        raise (Exception("=> no checkpoint found at '{}'".format(filename)))

    def load_state_dict(self, model, state_dict):
        if list(state_dict.keys())[0].split('.')[0] == "module":
            state_dict_temp = OrderedDict()
            for key, value in state_dict.items():
                state_dict_temp[".".join(key.split(".")[1:])] = value
            state_dict = state_dict_temp

        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
