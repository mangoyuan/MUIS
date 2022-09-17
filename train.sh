#!/bin/bash

ROOT=/path/to/save
CHECKPOINT=MUIS

# ----- Clustering -----
MODEL=clu96+256
CUDA_VISIBLE_DEVICES=0,1 python train_clu_mscale.py \
--model-name=${MODEL} \
--dataset-path1=${ROOT}/Edema96 \
--dataset-path2=${ROOT}/Edema256 \
--img-size=96 --aux-size=256 --checkpoint-root=${ROOT}/${CHECKPOINT}

IDX=000
CUDA_VISIBLE_DEVICES=0 python predict_mask_from_cam.py \
--checkpoint-root=${ROOT}/${CHECKPOINT} --model-name=${MODEL} --model-idx=${IDX} \
--dataset-path=${ROOT}/Edema96 --img-size=96 --target-size=256


# ------ Make Pseudo Segmentation labels -------------
THRESHOLD=0.1
CUDA_VISIBLE_DEVICES=0 python train_make_mscale_pseudo.py \
--checkpoint-root=${ROOT}/${CHECKPOINT} --model-name=${MODEL} --model-idx=${IDX} \
--dataset-path=${ROOT}/Edema256 --clu-input-size=96 --threshold=${THRESHOLD}

OMP_NUM_THREADS=1 python train_make_postp_pseudo.py \
--dataset-path=${ROOT}/Edema256 --pseudo-path=${ROOT}/${CHECKPOINT}/${MODEL}/${IDX}/mscalePseudoMasks${THRESHOLD}


# ----- Segmentation -----
CUDA_VISIBLE_DEVICES=0 python train_seg.py \
--dataset-path=${ROOT}/Edema256 --model-name=${MODEL}mseg --checkpoint-root=${ROOT}/${CHECKPOINT} \
--cluster-checkpoint=${ROOT}/${CHECKPOINT}/${MODEL}/${IDX} --beta-sc=0. --beta-aux=0. \
--pseudo-dir=${ROOT}/${CHECKPOINT}/${MODEL}/${IDX}/mscalePseudoMasks${THRESHOLD}post

CUDA_VISIBLE_DEVICES=0 python predict_mask_from_seg.py \
--checkpoint-root=${ROOT}/${CHECKPOINT} --model-name=${MODEL}seg --model-idx=${IDX} \
--dataset-path=${ROOT}/Edema256 --cluster-checkpoint=${ROOT}/${CHECKPOINT}/${MODEL}/${IDX}
