# main.py

import sys
import traceback
import torch
import torch.nn as nn
import random
import config
import utils
from model import Model
from train import Trainer
from test import Tester

from dataloader import Dataloader
from models import UnetSeg
import faulthandler
faulthandler.enable()

def main():
    # parse the arguments
    args = config.parse_args()
    if (args.ngpu > 0 and torch.cuda.is_available()):
        device = "cuda:0"
    else:
        device = "cpu"
    args.device = torch.device(device)
    args.rank = 0

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if args.save_results:
        utils.saveargs(args)

    # Create Model
    models = Model(args)
    model, criterion = models.setup(split_devices=False)

    # Occlusion segmentation model
    modelO = None
    modelO = UnetSeg(args, in_channels=3, out_channels=3)
    modelO.to(args.device)
    if args.ngpu > 1:
        modelO = nn.DataParallel(modelO, list(range(args.ngpu)))

    # Data Loading
    dataloader = Dataloader(args)

    # loaders = dataloader.create(shuffle=True)
    loaders = dataloader.create_subsets(shuffle=True)       # Only for CelebA

    # Initialize trainer and tester
    trainer = Trainer(args, model, modelO, criterion)
    tester = Tester(args, model, modelO, trainer.modelR, criterion, trainer.renderer, tb_writer=trainer.tb_writer)

    if args.eval:
        from eval import Evaluator
        evaluator = Evaluator(args, model, modelO, trainer.modelR, trainer.renderer, tb_writer=trainer.tb_writer)
        loaders = dataloader.create(shuffle=False)
        with torch.no_grad():
            evaluator.evaluate(0, loaders['train']) # change to test for celeba
        return

    # Run training/testing
    if args.test:
        loss_test = tester.test(0, test_loader)
        tester.tb_writer.close()
    else:
        # start training !!!
        loss_best = -1e10
        for epoch in range(trainer.epoch, args.nepochs):
            print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

            # train for a single epoch
            # loss_train = trainer.train(epoch, loaders)
            with torch.no_grad():  # operations inside don't track history
                loss_test = tester.test(epoch, loaders)

            if args.save_results:
                if loss_test > loss_best:   # > for psnr + 20*ssim
                    loss_best = loss_test
                    trainer.checkpoint.save_ckpt(epoch, trainer.ckpt_dict, prefix="checkpoint_recon", add_epoch=True)
                else:
                    trainer.checkpoint.save_ckpt(epoch, trainer.ckpt_dict, prefix="checkpoint_recon", add_epoch=False)
        trainer.tb_writer.close()


if __name__ == "__main__":
    main()
