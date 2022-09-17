# -*- coding: utf-8 -*-
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
import torchvision.transforms as ttf 

import data.transforms as dtf 


class EdemaWithMask(data.Dataset):
    def __init__(self, root, train, pseudo=None, json_name='ori1234.json', ram=True,
                 input_tsfm=None, mask_tsfm=None, joint_tsfm=None):
        super(EdemaWithMask, self).__init__()
        self.train = train 
        self.pseudo = pseudo
        self.ram = ram 
        self.json_name = json_name
        self.input_tsfm = input_tsfm
        self.mask_tsfm = mask_tsfm
        self.joint_tsfm = joint_tsfm
        self.data = self._load_list(root)
    
    def _load_list(self, root):
        with open(pjoin(root, self.json_name), 'r') as f:
            props = json.load(f)
        
        img_list = props['data']['train']
        img_list.extend(props['data']['val'])
        tic = time.time()
        m = self.pseudo if self.train else pjoin(root, 'masks')
        if self.ram:
            img_list = [(deepcopy(Image.open(pjoin(root, 'images', img+'.jpg'))),
                         deepcopy(Image.open(pjoin(m, img+'.png'))),
                         int(img[-1]), img)
                        for img in img_list]
        else:
            img_list = [(pjoin(root, 'images', img+'.jpg'),
                         pjoin(root, m, img+'.png'), int(img[-1]), img)
                        for img in img_list]
        print('Load images(train={}, ram={}) from {} and {}, which cost {:.4f}s'.format(
            self.train, self.ram, root, m, time.time() - tic))
        return img_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        img, mask, lbl, img_name = self.data[item]
        if not self.ram:
            img = Image.open(img); mask = Image.open(mask)
        if self.joint_tsfm is not None:
            img, mask = self.joint_tsfm(img, mask)
        if self.input_tsfm is not None:
            img = self.input_tsfm(img)
        if self.mask_tsfm is not None:
            mask = self.mask_tsfm(mask)
        return img, mask, lbl, img_name


def get_loader(root, train, pseudo_dir, size, bs, json_name='ori1234.json', ram=True):
    train_joint_tsfm = dtf.JointCompose([
        dtf.JointRotate(30),
        dtf.JointElasticDeform(sigmas=(9., 13.), points=3),
        dtf.JointRandomResizedCrop(size, scale=(0.4, 1.0), ratio=(3./4., 4./3.)),
        dtf.JointRandomHorizontalFip(p=0.5),
    ])

    input_tsfm = ttf.Compose([
        ttf.ToTensor(),
        ttf.Normalize(mean=[0.5], std=[0.5])
    ])
    train_joint_tsfm = None 
    val_joint_tsfm = None
    mask_tsfm = dtf.MaskToTensor()

    dataset = EdemaWithMask(root, train, pseudo_dir, json_name, ram,
                            joint_tsfm=train_joint_tsfm if train else None,
                            input_tsfm=input_tsfm, mask_tsfm=mask_tsfm)
    loader = data.DataLoader(dataset, bs, shuffle=train, num_workers=6, 
                             pin_memory=True, drop_last=train)
    return loader 
