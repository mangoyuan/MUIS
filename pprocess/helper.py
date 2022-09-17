# -*- coding: utf-8 -*-
""" Aha, some borrowed from nnUNet.
"""
import os
from os.path import join as pjoin
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
from skimage import transform
from collections import OrderedDict


def maybe_mkdir(path):
    if isinstance(path, tuple) or isinstance(path, list):
        for p in path:
            maybe_mkdir(p)
        return

    if not os.path.exists(path):
        os.makedirs(path)


def create_nonzero_mask(x):
    from scipy.ndimage import binary_fill_holes
    nonzero_mask = (x != 0)
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0, adjust_to=512, dataset_type='Edema'):
    if dataset_type == 'Edema':
        # For OCT-Edema. As the axis z and y always crop the full size, we only handle axis x.
        # Return bounding-box indexes according to the mask, adjust the length of x to [adjust_to].
        assert len(mask.shape) == 3, f'Expect mask has three dimension, but {mask.shape}!'
        z, x, y = mask.shape
        mask_voxel_coords = np.where(mask != outside_value)
        mean = np.average(mask_voxel_coords[1]).astype(int)
        up = max(mean - adjust_to // 2, 0)
        down = up + adjust_to
        return [[0, z], [up, down], [0, y]]
    else:
        raise NotImplemented


def crop_to_bbox(image, bbox):
    if len(image.shape) == 3:
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    elif len(image.shape) == 2:
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]))
    else:
        raise ValueError
    return image[resizer]


class EdemaReader(object):
    """ AIChallenge Edema Data Read Helper.
    :param root: path to `AIChallenger_edema`.
    :param smooth: gaussian smooth.
    :param single_label: ground truth may have multiple label, whether take all non-zero as single label.
    :param crop: boolean.
    :param adjust_to: adjust the #row (axis=x) of cropping.
    :return:
    """
    def __init__(self, root, smooth=False, single_label=False, crop=False, adjust_to=0):
        self.root = root
        self.img_set = dict(train=[], val=[])
        self._init_img_set()
        self.trainval = self.img_set['train'] + self.img_set['val']

        self.single_label = single_label
        self.crop = crop
        self.adjust_to = adjust_to
        self.smooth = smooth

    def _init_img_set(self):
        dirs = dict(train='Edema_trainingset', val='Edema_validationset')
        for phase, d in dirs.items():
            phase_path = pjoin(self.root, d, 'original_images')
            for case in sorted(os.listdir(phase_path)):
                case_path = pjoin(phase_path, case)
                self.img_set[phase].append(case_path)

    def load(self, idx, phase='trainval'):
        if phase in ('train', 'val'):
            path = self.img_set[phase][idx]
        elif phase == 'trainval':
            path = self.trainval[idx]
        else:
            raise ValueError

        images, masks = [], []
        n = len(os.listdir(path))
        for i in range(1, n+1):
            image_path = pjoin(path, f'{i}.bmp')
            mask_path = pjoin(path.replace('original_images', 'label_images').replace('.img', '_labelMark'),
                               f'{i}.bmp')

            image = Image.open(image_path)
            if self.smooth:
                image = image.filter(ImageFilter.SMOOTH)
            mask = Image.open(mask_path)

            image = np.array(image)
            mask = np.array(mask)
            images.append(image[None])
            masks.append(mask[None])

        images = np.concatenate(images, axis=0)  # z, x, y
        masks = np.concatenate(masks, axis=0)
        if self.single_label:
            masks[masks != 0] = 255
        if self.crop:
            bbox = get_bbox_from_mask(masks, adjust_to=self.adjust_to)
            images = crop_to_bbox(images, bbox)
            masks = crop_to_bbox(masks, bbox)
        # print(images.shape, masks.shape)
        return images, masks, path


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    root = '/path/to/AIChallenger_edema'
    reader = EdemaReader(root, True, True, False, 512)
    data = reader.load(10)
    img, msk, p = data
    fig, axes = plt.subplots(1, 2)

    index = 100
    axes[0].imshow(img[index])
    axes[1].imshow(msk[index])
    plt.title(os.path.basename(p))
    plt.show()

