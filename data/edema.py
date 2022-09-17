# -*- coding : utf-8 -*-

import os
from os.path import join as pjoin
import torch
import json
from PIL import Image
from torch.utils import data
import random
from copy import deepcopy
import sys
import time


class SingleScaleEdema(data.Dataset):
    def __init__(self, root, train, json_name='ori1234.json', ram=True):
        """ ram: load all image in RAM. """
        self.train = train
        self.ram = ram
        self.json_name = json_name
        self.data = self._load_list(root)

    def _load_list(self, root):
        with open(pjoin(root, self.json_name), 'r') as f:
            props = json.load(f)

        img_list = props['data']['train'] if self.train else props['data']['val']
        tic = time.time()
        if self.ram:
            img_list = [(deepcopy(Image.open(pjoin(root, 'images', img + '.jpg'))), int(img[-1]), img)
                        for img in img_list]
        else:
            img_list = [(pjoin(root, 'images', img + '.jpg'), int(img[-1]), img)
                        for img in img_list]
        print('Load images(train={}, ram={}) from {}, which cost {:.4f}s'.format(
            self.train, self.ram, root, time.time() - tic))
        return img_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, lbl, img_name = self.data[item]
        if not self.ram:
            img = Image.open(img)
        return img, lbl, img_name


class MultiScaleEdema(data.Dataset):
    def __init__(self, root1, root2, train, json_name='ori1234.json', ram=True):
        """ ram: load all image in RAM. """
        self.train = train
        self.ram = ram
        self.json_name = json_name
        self.data = self._load_list(root1, root2)

    def _load_list(self, root1, root2):
        with open(pjoin(root1, self.json_name), 'r') as f:
            props = json.load(f)

        img_list = props['data']['train'] if self.train else props['data']['val']
        tic = time.time()
        if self.ram:
            img_list = [(deepcopy(Image.open(pjoin(root1, 'images', img + '.jpg'))),
                         deepcopy(Image.open(pjoin(root2, 'images', img + '.jpg'))),
                         int(img[-1]), img)
                        for img in img_list]
        else:
            img_list = [(pjoin(root1, 'images', img + '.jpg'),
                         pjoin(root2, 'images', img + '.jpg'), int(img[-1]), img)
                        for img in img_list]
        # print(sys.getsizeof(img_list))
        print('Load images(train={}, ram={}) from {} and {}, which cost {:.4f}s'.format(
            self.train, self.ram, root1, root2, time.time() - tic))
        return img_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img1, img2, lbl, img_name = self.data[item]
        if not self.ram:
            img1 = Image.open(img1)
            img2 = Image.open(img2)
        return img1, img2, lbl, img_name


if __name__ == '__main__':
    pass
