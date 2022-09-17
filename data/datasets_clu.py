# -*- coding: UTF-8 -*-

import os
import random
from torch.utils.data import Dataset, ConcatDataset
import torchvision
import numpy as np
from PIL import Image


class ClusterDataset(Dataset):
    def __init__(self, args, training=True, multiscale=False):
        self.multiscale = multiscale
        self.training = training
        img_size = args.img_size

        if not multiscale:
            from data.edema import SingleScaleEdema
            root = args.dataset_path
            dataset_train = SingleScaleEdema(root, train=True, json_name=args.split_json)
            dataset_test = SingleScaleEdema(root, train=False, json_name=args.split_json)
        else:
            from data.edema import MultiScaleEdema
            root1, root2 = args.dataset_path1, args.dataset_path2
            dataset_train = MultiScaleEdema(root1, root2, train=True, json_name=args.split_json)
            dataset_test = MultiScaleEdema(root1, root2, train=False, json_name=args.split_json)
        self.dataset = ConcatDataset([dataset_train, dataset_test])
        self.data = dataset_train.data + dataset_test.data

        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.transforms_aug = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.transforms_aux = torchvision.transforms.Compose([  # auxiliary scale
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __getitem_sscale(self, item):
        img_raw, label, imn = self.dataset[item]
        img = self.transforms(img_raw)
        if self.training:
            img_aug = self.transforms_aug(img_raw)
            return img, img_aug, imn
        return img, label, imn

    def __getitem_mscale(self, item):
        img1_raw, img2_raw, label, imn = self.dataset[item]
        img1 = self.transforms(img1_raw)
        if self.training:
            img1_aug = self.transforms_aug(img1_raw)
            img2 = self.transforms_aux(img2_raw)
            return img1, img1_aug, img2, imn 
        return img1, label, imn

    def __getitem__(self, item):
        if not self.multiscale:
            return self.__getitem_sscale(item)
        return self.__getitem_mscale(item)

    def __len__(self):
        return len(self.dataset)
