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
from net import models
from data import datasets_clu as datasets


def predict(encoder, loader, target_size, checkpoint, fg_threshold, device):
    save_path = os.path.join(checkpoint, 'cluCamMask_{:.2f}'.format(fg_threshold))
    maybe_mkdir(save_path)
    print(f'[INFO] Saving to {save_path}')

    # predict and save as png.
    for data in tqdm(loader):
        img, _, imn = data
        img = img.to(device)
        b = img.size(0)

        _, zc_logit, _, cam, _ = encoder(img, return_cam=True)
        cam = F.interpolate(cam, size=(target_size, target_size), mode='bilinear', align_corners=False)
        zc = torch.argmax(zc_logit, dim=1)
        zc = zc.detach().cpu().numpy()

        max_v, _ = torch.max(cam.view(b, -1), dim=1, keepdim=True)
        norm_cam = cam / (max_v.view(b, 1, 1, 1) + 1e-5)
        norm_cam = norm_cam.cpu().numpy()

        for i in range(b):
            c = zc[i]
            m = norm_cam[i, c]
            prd = np.zeros_like(m)
            # re-scale [0, 1] to [127, 254] for visualization
            prd[m > fg_threshold] = (c + 1) * 127 

            prd = Image.fromarray(prd.astype(np.uint8))
            prd.save(pjoin(save_path, imn[i] + '.png'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-idx', type=str)

    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--img-size', type=int, default=96)
    parser.add_argument('--target-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--split-json', type=str, default='ori1234.json')
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument('--dim-zs', type=int, default=50)
    parser.add_argument('--dim-zc', type=int, default=2)

    parser.add_argument('--fg-threshold', type=float, default=0.3)

    args = parser.parse_args()
    checkpoint_path = os.path.join(args.checkpoint_root, args.model_name, args.model_idx)
    
    # build model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    en = models.get_encoder(args.dim_zs, args.dim_zc)
    model_path = os.path.join(checkpoint_path, 'model', 'last_encoder.tar')
    print(f'[INFO] Start load from {model_path}...')
    load_model(en, model_path)
    en.to(device)
    en.eval()

    # data loader
    test_dataset = datasets.ClusterDataset(args, training=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    with torch.no_grad():
        predict(en, test_loader, args.target_size, checkpoint_path, args.fg_threshold, device)
    print('Done!')
