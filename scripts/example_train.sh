#!/usr/bin/bash

export LD_LIBRARY_PATH=/scr/yzz/miniconda3/envs/vito/lib/

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
torchrun \
--nproc_per_node=1 \
--nnodes=1 \
scripts/train_tokenizer.py \
--config_file configs/example_config.json \
--default_root_dir "./" \
--log_every 10 \
--data_folder None \
--train_list metadata/k600_train_list.txt \
--val_list metadata/k600_val_list.txt \
--use_resize \
--use_crop \
--frames_per_clip 8 \
--stride 2 \
--sample_clip
