#!/bin/sh
if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore train_fullbody_512.py \
        --outdir ./training-runs-fullbody \
        --data /datazy/Datasets/UPT_512_320 \
        --gpus 8 --cfg fashion \
        --cond true --batch 24 --l1_weight 10 --seed 1 \
        --vgg_weight 20 --use_noise_const_branch True \
        --workers 4 --contextual_weight 0 --pl_weight 0 \
        --mask_weight 30
fi