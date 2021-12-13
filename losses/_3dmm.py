# regression.py

import torch
from torch import nn
import numpy as np

__all__ = ['_3DMM']

class _3DMM(object):
    """docstring for 3DMM"""
    def __init__(self):
        super(_3DMM, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.eps = 1e-8

    def norm_loss(self, predictions, labels, mask=None, conf=None, loss_type='l1', reduce_mean=True, p=1, viz=None):
        assert (loss_type in ['l1', 'l2', 'l2,1']), "Suporting loss type is ['l1', 'l2', 'l2,1']"

        inputs, targets = predictions, labels

        if loss_type == 'l1':
            loss = self.l1_loss(inputs, targets)
        elif loss_type == 'l2':
            loss = self.mse_loss(inputs, targets)
        elif loss_type == 'l2,1':
            diff = inputs - targets
            loss = torch.sqrt(torch.sum((diff ** 2) + 1e-16, dim=1, keepdim=True))
            if p != 1:
                loss = torch.pow(loss, p)

        if conf is not None:
            loss = loss.mean(dim=1, keepdim=True)
            # loss = loss *2**0.5 / (conf + self.eps) + (conf + self.eps).log()
            loss = (loss * torch.exp(-conf) + conf) / 2

        if mask is not None:
            loss = loss * mask * (np.prod([*mask.shape]) / (mask.sum() + self.eps))

        return loss.mean()

    def gan_loss(self, loss_map, mask=None, conf=None):
        if conf is not None:
            loss_map = loss_map.mean(dim=1, keepdim=True)
            loss_map = (loss_map * torch.exp(-conf) + conf) / 2

        if mask is not None:
            loss_map = loss_map * mask * (np.prod([*mask.shape]) / (mask.sum() + self.eps))

        return loss_map.mean()

