#!/bin/sh
if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore test.py \
    --dataroot test_datas --testtxt test_pairs.txt \
    --network checkpoints/pasta-gan++/network-snapshot-004408.pkl \
    --outdir test_results/upper \
    --batchsize 1 --testpart upper \
    --use-sleeve-mask
elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore test.py \
    --dataroot test_datas --testtxt test_pairs.txt \
    --network checkpoints/pasta-gan++/network-snapshot-004408.pkl \
    --outdir test_results/lower \
    --batchsize 1 --testpart lower \
    --use-sleeve-mask
elif [ $1 == 3 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore test.py \
    --dataroot test_datas --testtxt test_pairs.txt \
    --network checkpoints/pasta-gan++/network-snapshot-004408.pkl \
    --outdir test_results/full \
    --batchsize 1 --testpart full \
    --use-sleeve-mask
fi