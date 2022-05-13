#!/bin/bash


# python main.py \
# 	--mode train \
# 	--order instance \
# 	--model reservoir_simclr \
# 	--stm_size 128 \
# 	--stm_batch_size 64 \
# 	--stm_span 500 \
# 	--batch_size 128 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.15 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/r_simclr_span500

# python main.py \
# 	--mode train \
# 	--order instance \
# 	--model reservoir_simclr \
# 	--stm_size 128 \
# 	--stm_batch_size 64 \
# 	--stm_span 1000 \
# 	--batch_size 128 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.15 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/r_simclr_span1000

# python main.py \
# 	--mode train \
# 	--order instance \
# 	--model reservoir_simclr \
# 	--stm_size 128 \
# 	--stm_batch_size 64 \
# 	--stm_span 3000 \
# 	--batch_size 128 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.15 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/r_simclr_span3000

# python main.py \
# 	--mode train \
# 	--order instance \
# 	--model reservoir_simclr \
# 	--stm_size 128 \
# 	--stm_batch_size 64 \
# 	--stm_span 2000 \
# 	--batch_size 128 \
# 	--epochs 10 \
# 	--warmup_epochs 1 \
# 	--learning_rate_weights 0.15 \
# 	--learning_rate_biases 0.005 \
# 	--lr_decay 1 \
# 	--print_freq 400 \
# 	--save_freq 5 \
# 	--projector 2048-2048 \
# 	--save_dir ../models/r_simclr_span2000



python main.py \
  --mode train \
	--order instance \
	--model sliding_simclr \
	--batch_size 128 \
	--epochs 8 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/instance_simclr_b128

python finetune.py \
	--batch_size 256 \
	--epochs 50 \
	--learning_rate 0.2 \
	--lr_decay 0.1 \
	--decay_epochs 20 42 \
	--weight_decay 1e-4 \
	--eval_freq 5 \
	--ckpt_dir ../models/instance_simclr_b128



