#!/usr/bin/env bash

python pono_train.py --epochs 10 \
                --net_name aod-xavier \
                --lr 1e-4 \
                --use_gpu true \
                --gpu 3 \
                --ori_data_path data/images/ \
                --haze_data_path data/data/ \
                --val_ori_data_path data/images/ \
                --val_haze_data_path data/val/ \
                --num_workers 2 \
                --batch_size 8 \
                --val_batch_size 16 \
                --print_gap 500 \
                --model_dir ponomodels \
                --log_dir ponologs \
                --sample_output_folder ponosamples
