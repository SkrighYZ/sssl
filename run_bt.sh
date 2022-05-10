#!/bin/bash


python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 2 \
	--stm_batch_size 1 \
	--stm_span 1500 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_s2_b128_sb1 \
	--corrupt_rate 0.1 
#	--use_boundary

python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 8 \
	--stm_batch_size 4 \
	--stm_span 1500 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_s8_b128_sb4 \
	--corrupt_rate 0.1 
#	--use_boundary

python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 32 \
	--stm_batch_size 16 \
	--stm_span 1500 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_s32_b128_sb16 \
	--corrupt_rate 0.1 
#	--use_boundary

python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 256 \
	--stm_batch_size 128 \
	--stm_span 1500 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_s256_b128_sb128 \
	--corrupt_rate 0.1 
#	--use_boundary

# python main.py \
#   --mode train \
# 	--order iid \
# 	--model sliding_bt \
# 	--batch_size 128 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.15 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/iid_bt_b128



# python main.py \
# 	--order instance \
# 	--model sliding_bt \
# 	--batch_size 256 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.3 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 1 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/instance_bt

