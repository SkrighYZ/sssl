#!/bin/bash

# python main.py \
# 	--mode train \
# 	--order instance \
# 	--model reservoir_supervised \
# 	--batch_size 128 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.1 \
# 	--learning_rate_biases 0.1 \
# 	--ltm_size 255 \
# 	--stm_size 1 \
# 	--stm_span 500 \
# 	--lr_decay 1 \
# 	--weight_decay 1e-4 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--num_classes 51 \
# 	--save_dir ../models/reservoir_sup

# python main.py \
# 	--mode train \
# 	--order instance \
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
# 	--num_classes 51 \
# 	--save_dir ../models/instance_sup

python main.py \
	--mode train \
	--order iid \
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
	--num_classes 51 \
	--save_dir ../models/iid_sup