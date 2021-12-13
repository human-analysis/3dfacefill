# model.py

import math
import models
import losses
from torch import nn
import torch
import numpy as np

class Model:
    def __init__(self, args):
        self.args = args
        self.ngpu = args.ngpu
        self.device = args.device
        self.devices = []
        for idx in range(self.ngpu):
            self.devices.append(torch.device("cuda:{}".format(idx)))

        self.model_type = args.model_type
        self.loss_type = args.loss_type
        self.loss_options = args.loss_options

    def setup(self, split_devices=False):

        modelE = getattr(models, '_' + self.model_type.lower()).Encoder(self.args).to(self.device)
        modelDS = getattr(models, '_' + self.model_type.lower()).ShapeDecoder(self.args).to(self.device)
        modelDT = getattr(models, '_' + self.model_type.lower()).AlbedoDecoder(self.args).to(self.device)
        all_models = [modelE, modelDS, modelDT]
        criterion = getattr(losses, self.loss_type)(**self.loss_options)

        return all_models, criterion
