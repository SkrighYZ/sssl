#!/bin/bash


python main.py \
	--mode train \
	--order instance \
	--model reservoir_simclr \
	--ltm_size 128 \
	--stm_size 128 \
	--stm_span 1500 \
	--batch_size 128 \
	--stm_batch_size 64 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_simclr_s128_b128_sb64 \
	--corrupt_rate 0.1 
#	--use_boundary

# python main.py \
#   --mode train \
# 	--order iid \
# 	--model sliding_simclr \
# 	--batch_size 256 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.3 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/iid_simclr_b256


