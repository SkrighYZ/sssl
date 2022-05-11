python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 128 \
	--stm_batch_size 64 \
	--stm_span 2000 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_span2000

python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 128 \
	--stm_batch_size 64 \
	--stm_span 3000 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_span3000

python main.py \
	--mode train \
	--order instance \
	--model reservoir_bt \
	--stm_size 128 \
	--stm_batch_size 64 \
	--stm_span 4000 \
	--batch_size 128 \
	--epochs 10 \
	--warmup_epochs 1 \
	--learning_rate_weights 0.15 \
	--learning_rate_biases 0.005 \
	--lr_decay 1 \
	--print_freq 400 \
	--save_freq 5 \
	--projector 2048-2048 \
	--save_dir ../models/r_bt_span4000