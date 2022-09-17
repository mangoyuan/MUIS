# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import random
from elasticdeform import deform_random_grid


# For single inputs.
class MaskToTensor(object):
    def __call__(self, x):
        x = np.array(x)
        if np.max(x) > 100:
            x[x != 0] = 1
        x = torch.from_numpy(x).long()
        return x
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '()\n'
        return format_string


# For two inputs.
class JointRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3./4., 4./3.),
                 interpolation=Image.BILINEAR):
        super(JointRandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img, msk):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
               F.resized_crop(msk, i, j, h, w, self.size, Image.NEAREST)


class JointRandomHorizontalFip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(JointRandomHorizontalFip, self).__init__(p)

    def __call__(self, img, msk):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(msk)
        return img, msk


class JointRotate(transforms.RandomRotation):
    def __init__(self, degrees, reample=False, expand=False, center=None):
        super(JointRotate, self).__init__(degrees, reample, expand, center)

    def __call__(self, img, msk):
        angle = self.get_params(self.degrees)
        img_ = F.rotate(img, angle, Image.BILINEAR, self.expand, self.center)
        msk_ = F.rotate(msk, angle, Image.NEAREST, self.expand, self.center)
        return img_, msk_


class JointElasticDeform(object):
    def __init__(self, sigmas, points, p=0.5):
        self.sigmas = sigmas
        self.points = points
        self.p = p

    @staticmethod
    def get_params(sigmas):
        s = random.uniform(sigmas[0], sigmas[1])
        return s

    def __call__(self, img, msk):
        s = self.get_params(self.sigmas)
        if random.random() < self.p:
            img = np.array(img) / 255.; msk = np.array(msk)
            img, msk = deform_random_grid([img, msk], sigma=s, points=self.points, order=[0, 0])
            img *= 255
            img = Image.fromarray(img.astype(np.uint8)); msk = Image.fromarray(msk.astype(np.uint8))
        return img, msk

    def __repr__(self):
        format_string = self.__class__.__name__ + '(sigma={0}, points={1}, p={2})'.format(
            self.sigmas, self.points, self.p)
        return format_string


class JointCompose(transforms.Compose):
    def __init__(self, transforms):
        super(JointCompose, self).__init__(transforms)

    def __call__(self, img, msk):
        for t in self.transforms:
            img, msk = t(img, msk)
        return img, msk

