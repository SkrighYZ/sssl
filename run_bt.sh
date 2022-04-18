#!/bin/bash


python main.py \
	--order instance \
	--model reservoir_bt \
	--ltm_size 8 \
	--stm_size 8 \
	--stm_span 20 \
	--batch_size 16 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.3 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 1 \
	--projector 2048-2048 \
	--save_dir ../models/instance_bt

# python main.py \
# 	--order iid \
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
# 	--save_dir ../models/iid_bt



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

