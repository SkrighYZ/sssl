#!/bin/bash


python main.py \
	--order iid \
	--model supervised \
	--batch_size 256 \
	--buffer_size 256 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.2 \
	--learning_rate_biases 0.2 \
	--lr_decay 1 \
	--weight_decay 1e-4 \
	--print_freq 400 \
	--save_freq 1 \
	--projector 2048-2048 \
	--num_classes 51 \
	--save_dir ../models/iid_sup