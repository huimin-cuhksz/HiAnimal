#!/usr/bin/env bash

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=2 python -m apps.predict_single --input test_cases/1_cat/cat_normal.png --checkpoint checkpoints/netG_latest --output test_results/cat/cat_prediction --resolution 512
