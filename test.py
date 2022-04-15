import os
import argparse
import json
import time
import numpy as np
import math
import random
import sys
from pathlib import Path
import pickle

from torch import optim, nn
import torch

from models import BarlowTwins, SimCLR, ResNet18

from loading_utils import get_stream_data_loaders

def adjust_learning_rate(args, optimizer, loader, step):
	max_steps = args.epochs * len(loader)
	warmup_steps = args.warmup_epochs * len(loader)
	base_lr = 1
	if step < warmup_steps:
		lr = step / warmup_steps
	else:
		step -= warmup_steps
		max_steps -= warmup_steps
		q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
		#end_lr = base_lr * 0.001
		end_lr = base_lr * args.lr_decay
		lr = base_lr * q + end_lr * (1 - q)
	optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
	optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases

def train(args, model, device='cuda:0'):

	model.to(device)

	param_weights = []
	param_biases = []
	for param in model.parameters():
		if param.ndim == 1:
			param_biases.append(param)
		else:
			param_weights.append(param)
	parameters = [{'params': param_weights, 'lr': args.learning_rate_weights}, 
				{'params': param_biases, 'lr': args.learning_rate_biases}]
	optimizer = optim.SGD(parameters, lr=args.learning_rate_weights, momentum=args.momentum, weight_decay=args.weight_decay)

	dataset, train_loader, replay_loader, replay_sampler = get_stream_data_loaders(args)

	if args.model == 'sliding_supervised':
		criterion = nn.CrossEntropyLoss().to(device)

	model.train()
	start_time = time.time()
	start_epoch = 0
	
	for epoch in range(start_epoch, args.epochs):

		dataset.shuffle()
		if replay_sampler:
			replay_sampler.rehearsal_ixs = list(range(args.batch_size-1))
			replay_iter = iter(replay_loader)

		loss_total = 0

		for step, ((y1, y2), labels) in enumerate(train_loader, start=epoch*len(train_loader)):

			# pickle.dump(y1, open('../y1.pkl', 'wb'))
			# pickle.dump(y2, open('../y2.pkl', 'wb'))
			# break
			
			if replay_sampler:

				if step < args.batch_size-1:
					continue

				# Update sliding window buffer
				if step < args.buffer_size:
					replay_sampler.rehearsal_ixs += [step]
				else:
					replay_sampler.rehearsal_ixs = replay_sampler.rehearsal_ixs[1:] + [step]

				if (step + 1) % args.batch_size != 0:
					continue

				(replay_y1, replay_y2), _ = next(replay_iter)
				y1_inputs = torch.cat([y1.cuda(non_blocking=True), replay_y1.cuda(non_blocking=True)], dim=0)
				y2_inputs = torch.cat([y1.cuda(non_blocking=True), replay_y2.cuda(non_blocking=True)], dim=0)

			else:
				y1_inputs = y1.cuda(non_blocking=True)
				y2_inputs = y2.cuda(non_blocking=True)

			adjust_learning_rate(args, optimizer, train_loader, step)
			optimizer.zero_grad()

			if args.model == 'sliding_supervised':
				loss = criterion(model(y1_inputs), labels.cuda(non_blocking=True))
			else:
				loss = model(y1_inputs, y2_inputs)

			loss.backward()
			optimizer.step()

			loss_total += loss.item()

			if (step+1) % args.print_freq == 0:
				stats = dict(epoch=epoch,
							step=step,
							total=args.epochs*len(train_loader), 
							lr=optimizer.param_groups[0]['lr'],
							loss=loss.item(),
							time=int(time.time() - start_time))
				print(json.dumps(stats))

		if (epoch+1) % args.save_freq == 0:
			state = dict(epoch=epoch, step=step, model=model.state_dict(), optimizer=optimizer.state_dict())
			torch.save(state, args.save_dir / ('checkpoint-'+str(epoch)+'.pth'))

		stats = dict(epoch=epoch,
				lr=optimizer.param_groups[0]['lr'],
				avg_loss=loss_total/len(train_loader),
				time=int(time.time() - start_time))
		loss_total = 0
		print(json.dumps(stats))
		print('-----------------')

	torch.save(model.backbone.state_dict(), args.save_dir / 'resnet18.pth')

	return model



def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default='stream51', choices=['stream51'])
	parser.add_argument('--num_classes', type=int, default=51)
	parser.add_argument('--order', type=str, default='iid', choices=['iid', 'instance'])
	parser.add_argument('--model', type=str, default='sliding_bt',
						choices=['sliding_bt', 'reservoir_bt', 'cluster_bt', 'sliding_simclr', 'hnm_simclr', 'sliding_supervised'])

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--buffer_size', type=int, default=256)

	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--warmup_epochs', type=int, default=1)
	parser.add_argument('--learning_rate_weights', type=float, default=0.3)
	parser.add_argument('--learning_rate_biases', type=float, default=0.005)
	parser.add_argument('--lr_decay', type=float, default=1)
	parser.add_argument('--weight_decay', type=float, default=1e-6)
	parser.add_argument('--momentum', default=0.9, type=float)

	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--print_freq', default=400, type=int, metavar='N', help='print frequency')
	parser.add_argument('--save_freq', default=5, type=int, metavar='N', help='save frequency')

	parser.add_argument('--projector', default='2048-2048', type=str, metavar='MLP', help='projector MLP')
	parser.add_argument('--lambd', default=0.005, type=float, metavar='L', help='weight on off-diagonal terms')

	parser.add_argument('--images_dir', type=Path, metavar='DIR', default='../data/Stream-51')
	parser.add_argument('--save_dir', type=Path, metavar='DIR')

	args = parser.parse_args()
	print(args)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if args.model == 'sliding_bt':
		model = BarlowTwins(args)
	elif args.model == 'sliding_supervised':
		model = ResNet18(args)
	else:
		raise NotImplementedError('Model not supported.')

	# perform streaming classification
	train(args, model)


if __name__ == '__main__':
	main()