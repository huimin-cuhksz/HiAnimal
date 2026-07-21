#!/usr/bin/env bash

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=2 python -m apps.train_refine --dataroot training_examples/36views/ --batch_size 8 --num_threads 2
