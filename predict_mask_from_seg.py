# -*- coding: utf-8 -*-
"""
Thresholding the CAM and save as png.
"""
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
from skimage import transform
import yaml

from pprocess.helper import maybe_mkdir
from data.utils import get_gtmask_volumes
from misc.utils import load_model
from net import models_seg, models
from data import datasets_seg as datasets


def predict(encoder, decoder, loader, checkpoint, device, C=None):
    save_path = os.path.join(checkpoint, 'segMask')
    maybe_mkdir(save_path)

    # predict and save as png.
    for data in tqdm(loader):
        img, _, _, imn = data
        img = img.to(device)

        if C is not None:
            img_for_zc = F.interpolate(img, size=(96, 96), mode='bilinear', align_corners=False)
            _, zc_logit, _, _, _ = C(img_for_zc)
            zc = torch.argmax(zc_logit, dim=1)
            zc = zc.cpu().numpy()

        _, _, _, _, bottle, skips = encoder(img)
        pred, _ = decoder(bottle, skips)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy()

        for i in range(img.size(0)):
            prd = pred[i]
            temp = np.zeros_like(prd)
            if C is not None:
                zc_ = zc[i]
                temp[prd == (zc_ + 1)] = zc_ + 1
                prd = temp * 127
            else:
                prd *= 127
            prd = Image.fromarray(prd.astype(np.uint8))
            prd.save(pjoin(save_path, imn[i] + '.png'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-idx', type=str)

    parser.add_argument('--dim-zs', type=int, default=50, help='dimension of zs')
    parser.add_argument('--dim-zc', type=int, default=2, help='dimension of zc')

    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--split-json', type=str, default='ori1234.json')
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument('--cluster-checkpoint', type=str, default='')

    args = parser.parse_args()
    checkpoint_path = os.path.join(args.checkpoint_root, args.model_name, args.model_idx)

    # build model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    en = models_seg.get_encoder(args.dim_zs, args.dim_zc)
    de = models_seg.get_decoder()

    model_path = os.path.join(checkpoint_path, 'model', 'last_encoder.tar')
    print(f'[INFO] Start load segmentation model from {model_path}...')
    load_model(en, model_path)
    en.to(device)
    en.eval()
    
    model_path = os.path.join(checkpoint_path, 'model', 'last_decoder.tar')
    load_model(de, model_path)
    de.to(device)
    de.eval()

    if len(args.cluster_checkpoint) != 0:
        C = models.get_encoder(args.dim_zs, args.dim_zc)
        model_path = os.path.join(args.cluster_checkpoint, 'model', 'last_encoder.tar')
        print(f'[INFO] Start load clustering model from {model_path}...')
        load_model(C, model_path)
        C.to(device)
        C.eval()
    else:
        C = None

    # data loader
    eval_loader = datasets.get_loader(args.dataset_path, False, None, args.img_size, args.batch_size, 
                                      json_name=args.split_json, ram=True)

    with torch.no_grad():
        predict(en, de, eval_loader, checkpoint_path, device, C)
    print('Done!')
