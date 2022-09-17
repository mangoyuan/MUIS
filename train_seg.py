# -*- coding: UTF-8 -*-

import os
import time
import shutil
import random
import yaml
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from itertools import chain
import torchvision
from medpy.metric import binary

from misc import utils, metrics
from net import models_seg as models
from data import datasets_seg as datasets
from data.utils import get_gtmask_volumes


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str, required=True, help='path to the dataset')
    parser.add_argument('--cluster-checkpoint', type=str)
    parser.add_argument('--pseudo-dir', type=str)
    parser.add_argument('--checkpoint-root', type=str, default='./checkpoint', help='path to the checkpoint root')
    parser.add_argument('--model-name', type=str, default='DCCS', help='name of the model')

    parser.add_argument('--beta-sc', type=float, default=5.)
    parser.add_argument('--beta-aux', type=float, default=5.)
    parser.add_argument('--aux-size', type=int, default=96)

    parser.add_argument('--dim-zs', type=int, default=50, help='dimension of zs')
    parser.add_argument('--dim-zc', type=int, default=2, help='dimension of zc')

    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--lr-gamma', type=float, default=0.95)
    parser.add_argument('--lr-step', type=int, default=1)

    parser.add_argument('--seed', type=int, default=8888, help='random seed')
    parser.add_argument('--num-workers', type=int, default=6, help='number of workers for the dataloaders')
    
    parser.add_argument('--split-json', type=str, default='ori1234.json')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--clu-size', type=int, default=96)

    args = parser.parse_args()

    # create checkpoint directory
    # checkpoint_root/model_name/
    checkpoint_path = os.path.join(args.checkpoint_root, args.model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    model_idx = len(os.listdir(checkpoint_path))
    checkpoint_path = os.path.join(checkpoint_path, '%03d' % model_idx)
    os.makedirs(checkpoint_path)
    # directory to save models
    os.makedirs(os.path.join(checkpoint_path, 'model'), exist_ok=True)
    # directory to save code
    shutil.copytree(os.getcwd(), os.path.join(checkpoint_path, 'code'))

    # create logger
    console_logger, file_logger = utils.create_logger(os.path.join(checkpoint_path, 'train.log'))

    file_logger.info('Args: %s' % str(args))
    file_logger.info('Checkpoint path: %s' % checkpoint_path)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create datasets
    train_loader = datasets.get_loader(args.dataset_path, True, args.pseudo_dir, args.img_size, args.batch_size, 
                                       json_name=args.split_json, ram=True)
    eval_loader = datasets.get_loader(args.dataset_path, False, None, args.img_size, args.batch_size, 
                                      json_name=args.split_json, ram=True)

    # create models
    encoder = models.get_encoder(args.dim_zs, args.dim_zc)
    decoder = models.get_decoder()
    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            encoder = nn.DataParallel(encoder)
            decoder = nn.DataParallel(decoder)
        file_logger.info('Using %d GPU' % num_gpus)
    else:
        device = torch.device('cpu')
        file_logger.info('Using CPU')
    encoder.to(device)
    decoder.to(device)

    if args.cluster_checkpoint is not None:
        utils.load_model(encoder, os.path.join(args.cluster_checkpoint, 'model', 'last_encoder.tar'))
        file_logger.info(f'Load init weights from {args.cluster_checkpoint}...')
    
    # clustering confuse list and match
    with open(os.path.join(args.cluster_checkpoint, 'train.log'), 'r') as f:
        for line in f.readlines():
            if line[0] != '[':
                continue
            match = eval(line.strip())
    seg_match = [0] * (len(match) + 1)
    for pred, gt in match:
        seg_match[gt + 1] = pred + 1
    
    confuse_path = os.path.join(args.cluster_checkpoint, 'model', 'last_confuses.yaml')
    with open(confuse_path, 'r') as f:
        clu_confuses = yaml.load(f, Loader=yaml.FullLoader)

    gt_volumes = get_gtmask_volumes(args.dataset_path)

    # create optimizers
    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # create SummaryWriter
    writer = SummaryWriter(logdir=os.path.join(checkpoint_path, 'runs'), comment='_' + args.model_name)

    max_acc = 0
    max_nmi = 0
    max_ari = 0
    max_dice = 0
    global_step = 0
    for epoch in range(args.epochs):
        # train
        global_step = train_epoch(train_loader, encoder, decoder, device,
                                  optimizer, epoch, global_step, file_logger, writer, args)

        # eval
        val_freq = 1
        if epoch % val_freq == 0 or epoch == args.epochs - 1:
            max_acc, max_nmi, max_ari, max_dice = eval_epoch(eval_loader, encoder, decoder, 
                                                   device, epoch, checkpoint_path, file_logger, writer,
                                                   (max_acc, max_nmi, max_ari, max_dice), args,
                                                   gt_volumes, clu_confuses, seg_match)
        scheduler.step()

    writer.close()
    print('Done!')


def train_epoch(train_loader, encoder, decoder, device, optimizer, epoch, global_step, file_logger, writer, args):
    train_data_time = utils.AverageMeter()
    train_batch_time = utils.AverageMeter()
    train_seg_loss = utils.AverageMeter()
    train_aux_loss = utils.AverageMeter()
    train_sc_loss = utils.AverageMeter()

    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    encoder.train()
    decoder.train()

    lr = optimizer.param_groups[0]['lr']
    print(f'lr: {lr}.')

    tic = time.time()
    for data in train_loader:
        train_data_time.update(time.time() - tic)

        img1, mask, _, _ = data
        img1 = img1.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        b = img1.size(0)
        _, _, _, _, bottle1, skips1 = encoder(img1)
        pred1, _ = decoder(bottle1, skips1)
        seg_loss1 = ce_loss(pred1, mask)

        en_loss = seg_loss1 

        optimizer.zero_grad()
        en_loss.backward()
        optimizer.step()

        train_seg_loss.update(seg_loss1.item(), n=b)

        train_batch_time.update(time.time() - tic)

        global_step += 1
        tic = time.time()

    file_logger.info('Epoch {0} (train):\t'
                     'data_time: {data_time.sum:.2f}s\t'
                     'batch_time: {batch_time.sum:.2f}s\t'
                     'seg_loss: {seg_loss.avg:.4f}\t'
                     'sc_loss: {sc_loss.avg:.4f}\t'
                     'ax_loss: {ax_loss.avg:.4f}\t'.format(
        epoch, data_time=train_data_time, batch_time=train_batch_time, seg_loss=train_seg_loss, sc_loss=train_sc_loss, ax_loss=train_aux_loss))

    return global_step


def eval_epoch(eval_loader, encoder, decoder, device, epoch, checkpoint_path, file_logger, writer,
               best_metrics, args, gt_volumes, confuses, match_order):
    max_acc, max_nmi, max_ari, max_dice = best_metrics

    eval_data_time = utils.AverageMeter()
    eval_batch_time = utils.AverageMeter()

    zc_logit = list()
    y_true = list()
    prd_volumes = dict()
    for k, v in gt_volumes.items():
        prd_volumes[k] = np.zeros_like(v)

    encoder.eval()
    decoder.eval()

    tic = time.time()
    with torch.no_grad():
        for data in eval_loader:
            eval_data_time.update(time.time() - tic)
            img, _, y_true_, imn = data
            img = img.to(device, non_blocking=True)

            # for clustring.
            img_for_zc = F.interpolate(img, size=(args.clu_size, args.clu_size), mode='bilinear', align_corners=False)
            zs_, zc_logit_, _, _, _, _ = encoder(img_for_zc)

            zc_logit.append(zc_logit_.cpu().numpy())
            y_true.append(y_true_.cpu().numpy())

            # for segmentation.
            _, _, _, _, bottle, skips = encoder(img)
            seg, _ = decoder(bottle, skips)
            seg = seg[:, match_order, :, :]  # re-order by match
            seg = torch.argmax(seg, dim=1)
            seg = seg.cpu().numpy()

            for b in range(img.size(0)):
                img_name = imn[b]
                prd = seg[b]
                c1, c2, index, _ = img_name.split('_')
                case = f'{c1}_{c2}'
                index = int(index)
                if confuses[img_name] == 1:
                    prd_volumes[case][index, prd == 2] = 1

            eval_batch_time.update(time.time() - tic)
            tic = time.time()

    zc_logit = np.concatenate(zc_logit, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # calculate metrics
    y_pred = np.argmax(zc_logit, axis=1)

    num_classes = zc_logit.shape[1]
    match = utils.hungarian_match(y_pred, y_true, num_classes)
    y_pred = utils.convert_cluster_assignment_to_ground_truth(y_pred, match)

    acc = metrics.accuracy(y_pred, y_true)
    nmi = metrics.nmi(y_pred, y_true)
    ari = metrics.ari(y_pred, y_true)

    dices = []
    for case in gt_volumes.keys():
        d = binary.dc(prd_volumes[case], gt_volumes[case])
        dices.append(d)
    dice = np.mean(dices)

    max_acc = max(max_acc, acc)
    max_nmi = max(max_nmi, nmi)
    max_ari = max(max_ari, ari)
    if dice > max_dice:
        utils.save_model(encoder, os.path.join(checkpoint_path, 'model', 'best_encoder.tar'))
        utils.save_model(decoder, os.path.join(checkpoint_path, 'model', 'best_decoder.tar'))
    max_dice = max(max_dice, dice)

    tic = time.time()

    utils.save_model(encoder, os.path.join(checkpoint_path, 'model', 'last_encoder.tar'))
    utils.save_model(decoder, os.path.join(checkpoint_path, 'model', 'last_decoder.tar'))
    eval_save_time = time.time() - tic

    file_logger.info('Epoch {0} (eval):\t'
                     'data_time: {data_time.sum:.2f}s\t'
                     'batch_time: {batch_time.sum:.2f}s\t'
                     'save_time: {save_time:.2f}s\t'
                     'acc: {acc:.2f}% ({max_acc:.2f}%)\t'
                     'nmi: {nmi:.4f} ({max_nmi:.4f})\t'
                     'ari: {ari:.4f} ({max_ari:.4f})\t'
                     'dice: {dice:.4f} ({max_dice:.4f})\t'.format(epoch,
                                                               data_time=eval_data_time, batch_time=eval_batch_time,
                                                               save_time=eval_save_time,
                                                               acc=acc, max_acc=max_acc, nmi=nmi, max_nmi=max_nmi,
                                                               ari=ari, max_ari=max_ari,
                                                               dice=dice, max_dice=max_dice))
    file_logger.info(str(match))
    writer.add_scalars('acc', {'val': acc}, epoch)
    writer.add_scalars('nmi', {'val': nmi}, epoch)
    writer.add_scalars('ari', {'val': ari}, epoch)
    writer.add_scalars('dice', {'val': dice}, epoch)

    return max_acc, max_nmi, max_ari, max_dice


if __name__ == '__main__':
    _main()
