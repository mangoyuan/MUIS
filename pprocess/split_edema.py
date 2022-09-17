# -*- coding: utf-8 -*-
import os
from os.path import join as pjoin
import random
import numpy as np
import json


def make_split(root, shuffle, seed, balance, save_name='split'):
    """
    :param root: path to the data which has a prototype json file.
    :param shuffle: shuffle cases and re-split train-val.
    :param margin: ignore #slice at the boundary of normal and abnormal.
    :param seed:
    :param balance:
    :param save_name:
    :return:
    """
    json_path = pjoin(root, 'props.json')
    split_file = dict(name='AIChallenge_Edema', shuffle=shuffle, seed=seed, balance=balance)
    with open(json_path, 'r') as f:
        props = json.load(f)

    img_list_per_case = props['data_per_case']
    cases = list(img_list_per_case['train'].keys()) + list(img_list_per_case['val'].keys())
    if shuffle:  # shuffle by case and restore.
        new_img_list_per_case = dict(train=dict(), val=dict())
        old_img_list_per_case = dict()
        old_img_list_per_case.update(img_list_per_case['train'])
        old_img_list_per_case.update(img_list_per_case['val'])

        n_train = len(img_list_per_case['train'])
        random.shuffle(cases)
        train_cases, val_cases = cases[:n_train], cases[n_train:]
        phase_cases = dict(train=train_cases, val=val_cases)
        for phase in ('train', 'val'):
            for case in phase_cases[phase]:
                new_img_list_per_case[phase][case] = old_img_list_per_case[case]
        img_list_per_case = new_img_list_per_case

    split_set = dict(train=[], val=[])
    ndict = dict(train=dict(normal=0, edema=0), val=dict(normal=0, edema=0))
    for phase in ('train', 'val'):
        for case in img_list_per_case[phase]:
            split_set[phase].extend(img_list_per_case[phase][case])

            labels = [int(img.replace('.bmp', '')[-1]) for img in split_set[phase]]
            n_edema = sum(labels)
            n_normal = len(labels) - n_edema
            ndict[phase]['normal'], ndict[phase]['edema'] = n_normal, n_edema

    if balance:  # if balance, shuffle despite case.
        for phase in ('train', 'val'):
            normal, edema = ndict[phase]['normal'], ndict[phase]['edema']
            minv = min(normal, edema)
            labels = [int(img.replace('.bmp', '')[-1]) for img in split_set[phase]]

            normal = [img for i, img in enumerate(split_set[phase]) if labels[i] == 0]
            edema = [img for i, img in enumerate(split_set[phase]) if labels[i] == 1]
            random.shuffle(normal)
            random.shuffle(edema)

            normal = normal[:minv]
            edema = edema[:minv]

            ndict[phase]['normal'], ndict[phase]['edema'] = len(normal), len(edema)

            new_split = list(sorted(normal + edema))
            split_set[phase] = new_split

    split_file['data'] = split_set
    split_file['n'] = ndict
    print(split_file['n'])
    with open(pjoin(root, f'{save_name}{seed}.json'), 'w') as f:
        json.dump(split_file, f, indent=4)
    print('Done!')


if __name__ == '__main__':
    shuffle = False  # use the original train-val split or re-split (shuffle) by case.
    seed = 1234
    balance = False
    name = 'ori'

    debug = False
    if debug:
        root = '/path/to/AIC_Edema96'
    else:
        root = '/path/to/save/Edema96'

    print(root)
    random.seed(seed)
    np.random.seed(seed)
    make_split(root, shuffle, seed, balance, save_name=name)
