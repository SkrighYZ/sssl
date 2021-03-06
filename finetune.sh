#!/bin/bash


python finetune.py \
	--batch_size 256 \
	--epochs 50 \
	--learning_rate 0.2 \
	--lr_decay 0.1 \
	--decay_epochs 20 42 \
	--weight_decay 1e-4 \
	--eval_freq 5 \
	--ckpt_dir ../models/instance_bt_b256

python finetune.py \
	--batch_size 256 \
	--epochs 50 \
	--learning_rate 0.2 \
	--lr_decay 0.1 \
	--decay_epochs 20 42 \
	--weight_decay 1e-4 \
	--eval_freq 5 \
	--ckpt_dir ../models/instance_sup_b256