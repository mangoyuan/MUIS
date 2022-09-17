# -*- coding: utf-8 -*-

import os
import numpy as np


def get_gtmask_volumes(dataset_path):
    mask_volume_path = os.path.join(dataset_path, 'maskVolumesNpy')
    if not os.path.exists(mask_volume_path):
        print(f'{mask_volume_path} not exist! Try to call function make_mask_volumes() in predict.py')
        make_gtmask_volumes(dataset_path)
        
    print(f'Load maskVolumesNpy from {dataset_path}...')
    data = {}
    for case in os.listdir(mask_volume_path):
        case_name = case.replace('.npy', '')
        case_path = os.path.join(mask_volume_path, case)
        volume = np.load(case_path)
        data[case_name] = volume
    return data


def make_gtmask_volumes(dataset_path):
    save_root = os.path.join(path, 'maskVolumesNpy')
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    props_json = os.path.join(path, 'props.json')
    with open(props_json, 'r') as f:
        props = json.load(f)['data_per_case']
    img_list_per_case = {}
    img_list_per_case.update(props['train']); img_list_per_case.update(props['val'])
    for case in tqdm(img_list_per_case.keys()):
        img_list = img_list_per_case[case]
        volume = []
        for img_name in img_list:
            img_path = os.path.join(path, 'masks', img_name+'.png')
            img = Image.open(img_path)
            img = np.array(img)
            img[img != 0] = 1
            volume.append(img[None])

        volume = np.concatenate(volume, axis=0)
        save_path = os.path.join(save_root, case+'.npy')
        np.save(save_path, volume)
    return
