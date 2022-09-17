# -*- coding: utf-8 -*-

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
import yaml 

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

from pprocess.helper import maybe_mkdir
from misc.utils import load_model
from data.utils import get_gtmask_volumes
from data import datasets_clu as datasets
from net.models import get_encoder


def multiscale_fusing(encoder, img, scale=(96, ), target_size=256):
    b, n = img.size(0), len(scale)
    zc_avg, cam_avg = torch.zeros(b, 2).cuda(), torch.zeros(b, 2, target_size, target_size).cuda()
    for s in scale:
        img = F.interpolate(img, size=(s, s), mode='bilinear', align_corners=False)
        _, zc, _, cam, _ = encoder(img, return_cam=True)
        cam = F.interpolate(cam, size=(target_size, target_size), mode='bilinear', align_corners=False)
        zc_avg += zc 
        cam_avg += cam  
    return zc_avg / n, cam_avg / n 


def densecrf_inference(img, cam, t=5):
    """
    :param img: numpy array, shape=(1, H, W)
    :param cam: numpy array, shape=(C, H, W)
    """
    c, h, w = cam.shape
    bg = 1. - np.max(cam, axis=0, keepdims=True)
    probs = np.concatenate((bg, cam))
    U = unary_from_softmax(probs)
    d = dcrf.DenseCRF2D(w, h, c+1)
    d.setUnaryEnergy(U)

    pairwise_en = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01, ), img=img, chdim=0)
    d.addPairwiseEnergy(pairwise_en, compat=1)
    Q = d.inference(t)
    Q = np.array(Q).reshape((c+1, h, w))
    return Q[1:]


def create_multiscale_pseudo_masks(encoder, test_loader, checkpoint, device, args):
    # pseudo mask dir
    pseudo_mask_dir = os.path.join(checkpoint, f'mscalePseudoMasks{args.threshold}')
    print(f'[INFO] Saving to {pseudo_mask_dir}')
    maybe_mkdir(pseudo_mask_dir)

    # load gt masks for evaluating dice scores.
    gt_volumes = get_gtmask_volumes(args.dataset_path)
    # predictions place holder for evaluation.
    prd_volumes = dict()
    for k, v in gt_volumes.items():
        prd_volumes[k] = np.zeros_like(v)

    # predict.
    for data in tqdm(test_loader):
        img, _, imn = data
        img = img.to(device)

        img_for_zc = F.interpolate(img, size=(args.clu_input_size, args.clu_input_size), mode='bilinear', align_corners=False)
        _, zc, _, _, _ = encoder(img_for_zc, return_cam=True)
        _, cam = multiscale_fusing(encoder, img, scale=args.scales, target_size=args.img_size)

        zc = torch.argmax(zc, dim=1).cpu().numpy()

        b = img.size(0)
        max_v, _ = torch.max(cam.view(b, -1), dim=1, keepdim=True)
        norm_cam = cam / (max_v.view(b, 1, 1, 1) + 1e-5)
        norm_cam = norm_cam.detach().cpu().numpy()

        img = img.detach().cpu().numpy()
        for i in range(b):
            prd_class = zc[i] 
            image, prob = img[i], norm_cam[i]
            prob = densecrf_inference(image, prob, t=1)

            case1, case2, index, _ = imn[i].split('_')
            case = f'{case1}_{case2}'
            index = int(index)

            temp = np.zeros(image.shape[1:])
            temp[prob[prd_class] > args.threshold] = prd_class + 1
            temp *= 127  # 127 for class 0, 254 for class 1.
            pseudo_mask = Image.fromarray(temp.astype(np.uint8))
            pseudo_mask.save(os.path.join(pseudo_mask_dir, imn[i]+'.png'))
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-idx', type=str)

    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--clu-input-size', type=int, default=96)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--scales', nargs='+', type=int, default=(96, 128, 192))
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--split-json', type=str, default='ori1234.json')
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--dim-zs', type=int, default=50)
    parser.add_argument('--dim-zc', type=int, default=2)

    parser.add_argument('--threshold', type=float, default=0.3)

    args = parser.parse_args()
    checkpoint_path = os.path.join(args.checkpoint_root, args.model_name, args.model_idx)
    model_path = os.path.join(checkpoint_path, 'model', 'last_encoder.tar')

    # Model.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    en = get_encoder(args.dim_zs, args.dim_zc)
    model_path = os.path.join(checkpoint_path, 'model', 'last_encoder.tar')
    print(f'[INFO] Start load from {model_path}...')
    load_model(en, model_path)
    en.to(device)
    en.eval()

    test_dataset = datasets.ClusterDataset(args, training=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    with torch.no_grad():
        create_multiscale_pseudo_masks(en, test_loader, checkpoint_path, device, args)
    print('Done!')
