#!/bin/sh
if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_upper \
    --batchsize 1
elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_lower \
    --batchsize 1
elif [ $1 == 3 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_full \
    --batchsize 1
elif [ $1 == 4 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/ZMO_dresses_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zmo_dresses \
    --batchsize 1
elif [ $1 == 5 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_outfit.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_outfit \
    --batchsize 1
elif [ $1 == 6 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_upper_tuckin \
    --batchsize 1
elif [ $1 == 7 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_upper_editing_sleevepants \
    --batchsize 1
elif [ $1 == 8 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-000801.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0520/zalora_v2_upper_v2 \
    --batchsize 1
####### 0521
elif [ $1 == 9 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/upper/deepfashion \
    --batchsize 1
elif [ $1 == 10 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/upper/mpv \
    --batchsize 1
elif [ $1 == 11 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/upper/zalando_v1 \
    --batchsize 1
elif [ $1 == 12 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/upper/zalando_v2 \
    --batchsize 1
elif [ $1 == 13 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/upper/zalora_v1 \
    --batchsize 1
elif [ $1 == 14 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/upper/zalora_v2 \
    --batchsize 1
### test full
elif [ $1 == 15 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/ZMO_dresses_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/zmo_dresses \
    --batchsize 1
elif [ $1 == 16 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/deepfashion \
    --batchsize 1
elif [ $1 == 17 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/mpv \
    --batchsize 1
elif [ $1 == 18 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/zalando_v1 \
    --batchsize 1
elif [ $1 == 19 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/zalando_v2 \
    --batchsize 1
elif [ $1 == 20 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/zalora_v1 \
    --batchsize 1
elif [ $1 == 21 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/full/zalora_v2 \
    --batchsize 1
### test lower
elif [ $1 == 22 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/lower/deepfashion \
    --batchsize 1
elif [ $1 == 23 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/lower/mpv \
    --batchsize 1
elif [ $1 == 24 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/lower/zalando_v1 \
    --batchsize 1
elif [ $1 == 25 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/lower/zalando_v2 \
    --batchsize 1
elif [ $1 == 26 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/lower/zalora_v1 \
    --batchsize 1
elif [ $1 == 27 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0521/lower/zalora_v2 \
    --batchsize 1
### 0524
# upper
elif [ $1 == 28 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/upper/deepfashion \
    --batchsize 1
elif [ $1 == 29 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/upper/mpv \
    --batchsize 1
elif [ $1 == 30 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/upper/zalando_v1 \
    --batchsize 1
elif [ $1 == 31 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/upper/zalando_v2 \
    --batchsize 1
elif [ $1 == 32 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/upper/zalora_v1 \
    --batchsize 1
elif [ $1 == 33 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/upper/zalora_v2 \
    --batchsize 1
### test lower
elif [ $1 == 34 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/lower/deepfashion \
    --batchsize 1
elif [ $1 == 35 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/lower/mpv \
    --batchsize 1
elif [ $1 == 36 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/lower/zalando_v1 \
    --batchsize 1
elif [ $1 == 37 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/lower/zalando_v2 \
    --batchsize 1
elif [ $1 == 38 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/lower/zalora_v1 \
    --batchsize 1
elif [ $1 == 39 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/lower/zalora_v2 \
    --batchsize 1
### test full
elif [ $1 == 40 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/ZMO_dresses_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/zmo_dresses \
    --batchsize 1
elif [ $1 == 41 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/deepfashion \
    --batchsize 1
elif [ $1 == 42 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/mpv \
    --batchsize 1
elif [ $1 == 43 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/zalando_v1 \
    --batchsize 1
elif [ $1 == 44 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/zalando_v2 \
    --batchsize 1
elif [ $1 == 45 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/zalora_v1 \
    --batchsize 1
elif [ $1 == 46 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-004408.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0524/full/zalora_v2 \
    --batchsize 1
### application
### tuckin tuckout ####
elif [ $1 == 47 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_app.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0526/zalora_v2_upper_tuckin \
    --batchsize 1
elif [ $1 == 48 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami_app.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0526/zalora_v2_upper_tuckinout \
    --batchsize 1
### garment editing ####
### tops
elif [ $1 == 49 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_app.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0526/zalora_v2_upper_garment_editing \
    --batchsize 1
### bottoms
elif [ $1 == 50 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_app.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0526/zalora_v2_bottoms_garment_editing \
    --batchsize 1
### reshape retexture
### tryon, reshape, retexture
elif [ $1 == 51 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_app.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0526/zalora_v2_upper_reshape_retexture \
    --batchsize 1
### suits tryon
elif [ $1 == 52 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_outfit.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0528/zalora_v2_full_outfits_v2 \
    --batchsize 1
#### lower debug 
elif [ $1 == 53 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami_app_debug.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/pastagan_apps_0528/zalora_v2_lower_debug/modify_patch \
    --batchsize 1
### lower
### test lower
elif [ $1 == 54 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/lower/all_pred \
    --batchsize 1
elif [ $1 == 55 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/lower/all_pred \
    --batchsize 1
elif [ $1 == 56 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/lower/all_pred \
    --batchsize 1
elif [ $1 == 57 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/lower/all_pred \
    --batchsize 1
elif [ $1 == 58 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/lower/all_pred \
    --batchsize 1
elif [ $1 == 59 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/lower/all_pred \
    --batchsize 1
### test full
elif [ $1 == 60 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/ZMO_dresses_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/dress_pred \
    --batchsize 1
elif [ $1 == 61 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/all_pred \
    --batchsize 1
elif [ $1 == 62 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/all_pred \
    --batchsize 1
elif [ $1 == 63 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/all_pred \
    --batchsize 1
elif [ $1 == 64 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/all_pred \
    --batchsize 1
elif [ $1 == 65 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/all_pred \
    --batchsize 1
elif [ $1 == 66 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/full/all_pred \
    --batchsize 1
## test upper
elif [ $1 == 67 ]; then
    CUDA_VISIBLE_DEVICES=2 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Deepfashion_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/upper/all_pred \
    --batchsize 1
elif [ $1 == 68 ]; then
    CUDA_VISIBLE_DEVICES=3 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/MPV_512_320 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/upper/all_pred \
    --batchsize 1
elif [ $1 == 69 ]; then
    CUDA_VISIBLE_DEVICES=4 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/upper/all_pred \
    --batchsize 1
elif [ $1 == 70 ]; then
    CUDA_VISIBLE_DEVICES=5 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalando_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/upper/all_pred \
    --batchsize 1
elif [ $1 == 71 ]; then
    CUDA_VISIBLE_DEVICES=6 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v1 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/upper/all_pred \
    --batchsize 1
elif [ $1 == 72 ]; then
    CUDA_VISIBLE_DEVICES=7 python3 -W ignore test_for_tpami.py \
    --dataroot /datazy/Datasets/UPT_512_320/Zalora_512_320_v2 \
    --network /datazy/Codes/PASTA-GAN-512_PAMI/training-runs-wo_flow-full-refinelowerpatch-0511/00001-UPT_512_320-cond-fashion-batch24-resumecustom/network-snapshot-001803.pkl \
    --outdir /datazy/Datasets/upt_results/upt_results_0529/upper/all_pred \
    --batchsize 1
fi