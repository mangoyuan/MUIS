# -*- coding: utf-8 -*-
""" Create pseudo masks for training segmentation network. """
import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import json
from PIL import Image
import pickle
import SimpleITK as sitk
from medpy.metric import dc
from multiprocessing import Pool 
import yaml 
import time

import skimage
from scipy.ndimage import label, binary_fill_holes, binary_closing
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
from imantics import Polygons, Mask 

from pprocess.helper import maybe_mkdir


def post_processing(img, msk, preserve_ratio=0.3):
    L = np.unique(msk)
    assert len(L) <= 2
    if len(L) < 2:
        return msk 

    L = L[1]
    foreground = np.copy(msk)
    foreground[foreground != 0] = 1
    foreground = binary_fill_holes(foreground)
    foreground = binary_closing(foreground)

    lmap, n = label(foreground.astype(int))
    areas = [0 for _ in range(n+1)]
    for i in range(1, n+1):
        areas[i] = np.sum(lmap == i)
    max_area = max(areas)
    for i in range(1, n+1):
        # ignore the region whose area lower than #preserve_ratio 
        if areas[i] < preserve_ratio * max_area:
            foreground[lmap == i] = 0
            areas[i] = 0

    polygons = Mask(foreground).polygons()

    new_mask = np.zeros_like(msk)
    for i, points in enumerate(polygons.points):
        rr, cc = polygon(points[:, 0], points[:, 1], new_mask.shape)
        temp_b = np.zeros_like(new_mask)  # before
        temp_b[cc, rr] = 1
        area_b = np.sum(temp_b == 1)

        snake = active_contour(gaussian(img, 3), snake=points, w_edge=3)

        rr, cc = polygon(snake[:, 0], snake[:, 1], new_mask.shape)
        temp_a = np.zeros_like(new_mask)  # after
        temp_a[cc, rr] = 1
        area_a = np.sum(temp_b == 1)

        # if the area is reduced to lower than 10%, preserve the original area.
        if area_a < 0.1 * area_b:
            new_mask[temp_b == 1] = L
        else:
            new_mask[temp_a == 1] = L
    return new_mask
    

def single_thread_postprocess_and_save(img_path, pse_path, save_path):
    img = Image.open(img_path); img = np.array(img)
    pse = Image.open(pse_path); pse = np.array(pse)
    pse_snake = post_processing(img, pse)
    pse_snake = Image.fromarray(pse_snake.astype(np.uint8))
    pse_snake.save(save_path)


def multi_thread_postprocess_and_save(img_root, pse_root, postfix, nthread=6):
    args = []
    maybe_mkdir(pse_root+postfix)
    for f in os.listdir(pse_root):
        if not f.endswith('.png'):
            continue
        f = f.split('.')[0]
        p1 = pjoin(img_root, 'images', f+'.jpg')
        p2 = pjoin(pse_root, f+'.png')
        p3 = pjoin(pse_root+postfix, f+'.png')
        arg = (p1, p2, p3)
        args.append(arg)
    p = Pool(nthread)
    p.starmap(single_thread_postprocess_and_save, args)
    p.close()
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--pseudo-path', type=str)
    parser.add_argument('--nthread', type=int, default=6)
    parser.add_argument('--postfix', type=str, default='post')

    print(f'[WARN] Version of skimage package must be lower than 0.18, now is {skimage.__version__}.')
    print(f'((/- -)/.... It may take near 18 minutes ....((/- -)/')
    args = parser.parse_args()
    tic = time.time()
    multi_thread_postprocess_and_save(args.dataset_path, args.pseudo_path, args.postfix, args.nthread)
    print(f'Done!, which cost {time.time() - tic}s.')
