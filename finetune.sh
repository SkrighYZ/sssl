#!/bin/bash


python finetune.py \
	--batch_size 256 \
	--epochs 80 \
	--learning_rate 0.3 \
	--lr_decay 0.1 \
	--decay_epochs 30 60 \
	--weight_decay 1e-4 \
	--eval_freq 10 \
	--ckpt_dir ../models/instance_bt