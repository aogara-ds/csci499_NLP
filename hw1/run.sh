#!/bin/bash

# Vanilla LSTM with word-level tokenization
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --batch_size=20000 \
    --num_epochs=10 \
    --val_every=1 \
    --force_cpu \
    --vocab_size=1000 \
    --show_plot 

# Vanilla LSTM with BPE tokenization
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --batch_size=20000 \
    --num_epochs=10 \
    --val_every=1 \
    --force_cpu \
    --vocab_size=1000 \
    --show_plot \
    --bpe

# LSTM with MaxPool and word-level tokenization
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --batch_size=20000 \
    --num_epochs=10 \
    --val_every=1 \
    --force_cpu \
    --vocab_size=1000 \
    --show_plot \
    --maxpool