#!/usr/bin/bash

export LD_LIBRARY_PATH=/scr/yzz/miniconda3/envs/vito/lib/

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. \
torchrun \
--nproc_per_node=4 \
--nnodes=1 \
scripts/train_tokenizer.py \
--config_file configs/vit_bb_k600.json \
--default_root_dir "output_dir/vit_bb_k600" \
--log_every 10 \
--data_folder None \
--train_list metadata/k600_train_list.txt \
--val_list metadata/k600_val_list.txt \
--use_resize \
--use_crop \
--frames_per_clip 8 \
--stride 2 \
--batch_size 8 \
--num_workers 8 \
--use_wandb \
--sample_clip
