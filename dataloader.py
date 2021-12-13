# dataloader.py

import torch
import datasets
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import albumentations
from albumentations.pytorch import ToTensor
from torch.utils.data.distributed import DistributedSampler

class Dataloader:
    def __init__(self, args):
        self.args = args
        self.dist = args.dist
        if self.dist:
            self.world_size = args.ngpu
            self.rank = args.rank

        self.dataset_options = args.dataset_options
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train

        self.train_dev_percent = args.train_dev_percent
        self.test_dev_percent = args.test_dev_percent
        self.resolution = args.resolution

        if self.dataset_train_name == 'Faces':
            self.dataset_train = datasets.Faces(
                args=args,
                root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='train'
            )

        elif hasattr(datasets, self.dataset_train_name):
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                args=self.args,
                root=self.args.dataroot,# + "/300W_LP",
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='train'
            )


        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == 'Faces':
            self.dataset_test = datasets.Faces(
                args=args,
                root=self.args.dataroot,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='test'
            )

        elif hasattr(datasets, self.dataset_test_name):
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                args=args,
                root=self.args.dataroot,# + "/300W_LP",
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                split='test'
            )

        else:
            # raise(Exception("Unknown Dataset"))
            return

    def create(self, flag=None, shuffle=True):
        dataloader = {}

        train_sampler = DistributedSampler(self.dataset_train, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle) if self.dist else None
        test_sampler = DistributedSampler(self.dataset_test, num_replicas=self.world_size, rank=self.rank, shuffle=False) if self.dist else None

        if flag is None:
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=(train_sampler is None and shuffle), pin_memory=True,
                sampler = train_sampler
            )
            dataloader['train_sampler'] = train_sampler

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True,
                sampler = test_sampler
            )
            dataloader['test_sampler'] = test_sampler

            dataloader['eval'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=(test_sampler is None and shuffle), pin_memory=True,
                sampler = test_sampler
            )
            return dataloader

        elif flag.lower() == "train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=(train_sampler is None and shuffle), pin_memory=True,
                sampler = train_sampler
            )
            dataloader['train_sampler'] = train_sampler
            return dataloader

        elif flag.lower() == "test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True,
                sampler = test_sampler
            )
            dataloader['test_sampler'] = test_sampler
            return dataloader

    def create_subsets(self, flag=None, shuffle=False):
        dataloader = {}

        if flag is None:
            train_len = len(self.dataset_train)
            train_cut_index = int(train_len * (self.train_dev_percent))
            indices = list(torch.arange(train_len))
            train_indices = indices[:train_cut_index]
            test_indices = indices[train_cut_index:]

            train_sampler = torch.utils.data.Subset(self.dataset_train, train_indices)
            test_sampler = torch.utils.data.Subset(self.dataset_train, test_indices)

            dataloader['train'] = torch.utils.data.DataLoader(
                train_sampler, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                test_sampler, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.nthreads,
                pin_memory=True
            )

            dataloader['eval'] = torch.utils.data.DataLoader(
                test_sampler, batch_size=self.args.batch_size,
                shuffle=True, num_workers=self.args.nthreads,
                pin_memory=True
            )

        elif flag.lower() == 'train':
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size,
                shuffle=shuffle, num_workers=self.args.nthreads,
                pin_memory=True
            )
        elif flag == "all":
            dataloader['all'] = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.nthreads,
                pin_memory=True
            )

        return dataloader
