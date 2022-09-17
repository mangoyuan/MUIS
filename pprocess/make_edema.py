# -*- coding: utf-8 -*-
import os
from os.path import join as pjoin
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import helper


def mkdataset(reader: helper.EdemaReader, save_path, new_size):
    image_path = pjoin(save_path, 'images')
    mask_path = pjoin(save_path, 'masks')
    helper.maybe_mkdir([save_path, image_path, mask_path])

    props = dict()
    props['name'] = 'AIChallenge_Edema'
    img_list_per_case = dict(train=dict(), val=dict())
    img_list = dict(train=[], val=[])

    n_normal = dict(train=0, val=0, all=0)
    n_total = dict(train=0, val=0, all=0)
    for phase in ('train', 'val'):
        for i in tqdm(range(len(reader.img_set[phase]))):
            image, mask, name = reader.load(i, phase=phase)
            base_name = os.path.basename(name)
            base_names = base_name.split('_')
            new_name = f'{base_names[0]}_{base_names[5]}'
            img_list_per_case[phase][new_name] = []

            for k in range(image.shape[0]):
                img, msk = image[k], mask[k]

                # for record.
                normal = int(np.sum(msk) != 0)
                if normal == 0:
                    n_normal[phase] += 1
                n_total[phase] += 1

                str_k = (3 - len(str(k))) * '0' + str(k)
                save_name = f'{new_name}_{str_k}_{normal}'
                image_save_path = pjoin(image_path, save_name + '.jpg')
                mask_save_path = pjoin(mask_path, save_name + '.png')
                assert save_name not in img_list[phase], f'{save_name} already exitst!'
                img_list[phase].append(save_name)
                img_list_per_case[phase][new_name].append(save_name)

                # for saving.
                h, w = img.shape
                img = Image.fromarray(img.astype(np.uint8))
                msk = Image.fromarray(msk.astype(np.uint8))
                if new_size != h or new_size != w:
                    img = img.resize((new_size, new_size), Image.BILINEAR)
                    msk = msk.resize((new_size, new_size), Image.NEAREST)

                img.save(image_save_path)
                msk.save(mask_save_path)

    n_normal['all'] = n_normal['train'] + n_normal['val']
    n_total['all'] = n_total['train'] + n_total['val']

    props['normal'] = n_normal
    props['total'] = n_total
    props['data'] = img_list
    props['data_per_case'] = img_list_per_case

    json_name = 'props.json'
    with open(pjoin(save_path, json_name), 'w') as f:
        json.dump(props, f, indent=4)
    print('Done!')


if __name__ == '__main__':
    single_label = True
    crop = True
    adjust_to = 96
    new_size = 96
    smooth = False

    debug = False
    if debug:
        root = '/path/to/AIChallenger_edema'
        save_path = f'/path/to/save/Test{new_size}'
    else:
        root = '/path/to/ai_challenger_fl2018'
        save_path = f'/path/to/save/Edema{new_size}'

    print(save_path)
    reader = helper.EdemaReader(root, smooth=smooth, single_label=single_label, crop=crop, adjust_to=adjust_to)
    mkdataset(reader, save_path, new_size)
