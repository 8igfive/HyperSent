#!/bin/bash

mkdir -p results/runs

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/230718/wiki1m_aigen_remove_20k_2.json \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 384 \
    --temp 0.05 \
    --num_layers 12 \
    --pooler_type cls \
    --hyperbolic_size 768 \
    --model_type bert \
    --output_dir results/runs/test_training_b \
    --hierarchy_type aigen \
    --disable_hyper \
    --hierarchy_levels 3\
    --overwrite_cache \
    --overwrite_output_dir \
    --do_train \
    --do_eval \