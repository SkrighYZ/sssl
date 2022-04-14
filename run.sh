#!/bin/bash

python main.py \
	--order iid \
	--model sliding-bt \
	--batch_size 256 \
	--epochs 50 \
	--warmup_epochs 5 \
	--learning_rate_weights 0.2 \
	--learning_rate_biases 0.005 \
	--print_freq 200 \
	--save-freq 10 \
	--projector 2048-2048 \
	--save_dir ../models/sliding-bt
