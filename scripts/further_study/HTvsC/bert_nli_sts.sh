#!/bin/bash

mkdir -p results/runs

python train.py \
    --model_name_or_path /home/LAB/limx/download/model/bert-base-uncased \
    --train_file data/230807/nliunsup_aigen_sts_3_20k.jsonl \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --temp 0.05 \
    --num_layers 12 \
    --pooler_type mask \
    --hyperbolic_size 768 \
    --model_type bert \
    --output_dir results/runs/further_study/HTvsC/bert_nli_sts_abs48_l2lastn \
    --hierarchy_type aigen \
    --disable_hyper \
    --hierarchy_levels 3\
    --overwrite_cache \
    --overwrite_output_dir \
    --do_train \
    --do_eval \