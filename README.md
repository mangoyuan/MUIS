# MUIS

Keywords: Unsupervised Segmentation, OCT, Scale-invariant.

## Overview

Source code of "Multiscale Unsupervised Retinal Edema Area Segmentation in OCT Images (MICCAI2022)".

The code will be released soon.

## Make Dataset 

1. Download `ai_challenger_fl2018` dataset from ai challenge.
2. Run below 
```bash
cd pprocess
# change the varibale `adjust_to` and `new_size` to 96
# and get the dataset with image size 96x96
python make_edema.py  
# split the train-test
python split_edema.py

# change the varibale `adjust_to` and `new_size` to 256
# and get the dataset with image size 256x256
python make_edema.py  
# reuse the split file in Edema96 dataset
```

## Train

```bash
./train.sh
```

## Thanks

- [DCCS](https://github.com/sKamiJ/DCCS)
