#!/bin/bash


python main.py \
	--mode train \
	--order instance \
	--model sliding_supervised \
	--batch_size 256 \
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
	--save_dir ../models/instance_sup

# python main.py \
# 	--mode train \
# 	--order iid \
# 	--model sliding_supervised \
# 	--batch_size 256 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.2 \
# 	--learning_rate_biases 0.2 \
# 	--lr_decay 1 \
# 	--weight_decay 1e-4 \
# 	--print_freq 400 \
# 	--save_freq 1 \
# 	--projector 2048-2048 \
# 	--num_classes 51 \
# 	--save_dir ../models/iid_sup