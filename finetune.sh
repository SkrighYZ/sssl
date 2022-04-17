#!/bin/bash


python finetune.py \
	--batch_size 256 \
	--epochs 100 \
	--learning_rate 0.3 \
	--lr_decay 0.1 \
	--decay_epochs 40 80 \
	--weight_decay 1e-4 \
	--eval_freq 10 \
	--ckpt_dir ../models/instance_sup