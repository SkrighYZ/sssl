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
from copy import deepcopy

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

	dataset, train_loader, replay_sampler = get_stream_data_loaders(args)

	if 'supervised' in args.model:
		criterion = nn.CrossEntropyLoss().to(device)

	model.train()
	start_time = time.time()
	start_epoch = 0

	loss_logs = []
	
	for epoch in range(start_epoch, args.epochs):

		dataset.shuffle()
		if replay_sampler is not None:
			print('Simulating batches...')
			replay_sampler.init_memory(ltm_size=args.ltm_size, stm_size=args.stm_size)
			replay_sampler.simulate_batches(ltm_size=args.ltm_size, stm_size=args.stm_size, batch_size=args.batch_size, num_examples=len(dataset))

		loss_total = 0
		for step, (y, labels) in enumerate(train_loader, start=epoch*len(train_loader)):

			if 'supervised' in args.model:
				inputs = y.cuda(non_blocking=True)
				targets = labels.cuda(non_blocking=True)
			else:
				y1, y2 = y
				y1_inputs = y1.cuda(non_blocking=True)
				y2_inputs = y2.cuda(non_blocking=True)

			adjust_learning_rate(args, optimizer, train_loader, step)
			optimizer.zero_grad()

			if 'supervised' in args.model:
				loss = criterion(model(inputs), targets)
			else:
				loss = model(y1_inputs, y2_inputs)

			loss.backward()
			optimizer.step()

			loss_total += loss.item()

			loss_logs.append(loss.item())

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
		loss_logs.append(loss_total/len(train_loader))
		loss_total = 0
		print(json.dumps(stats))
		print('-----------------')

	torch.save(model.backbone.state_dict(), args.save_dir / 'resnet18.pth')

	with open(args.save_dir / 'loss_logs.txt', 'w') as f:
		f.write('\n'.join([str(ll) for ll in loss_logs]))

	return model



def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', type=str, choices=['train', 'test'])

	parser.add_argument('--dataset', type=str, default='stream51', choices=['stream51'])
	parser.add_argument('--num_classes', type=int, default=51)
	parser.add_argument('--order', type=str, default='iid', choices=['iid', 'instance'])
	parser.add_argument('--model', type=str, default='sliding_bt',
						choices=['sliding_bt', 'reservoir_bt', 'sliding_supervised', 'reservoir_supervised'])

	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--ltm_size', type=int, default=128)
	parser.add_argument('--stm_size', type=int, default=128)
	parser.add_argument('--stm_span', type=int, default=1000)

	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--warmup_epochs', type=int, default=1)
	parser.add_argument('--learning_rate_weights', type=float, default=0.3)
	parser.add_argument('--learning_rate_biases', type=float, default=0.005)
	parser.add_argument('--lr_decay', type=float, default=1)
	parser.add_argument('--weight_decay', type=float, default=1e-6)
	parser.add_argument('--momentum', default=0.9, type=float)

	parser.add_argument('--num_workers', type=int, default=8)
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

	with open(args.save_dir / 'args.txt', 'w') as f:
		args_copy = deepcopy(args.__dict__)
		args_copy['images_dir'] = str(args.images_dir)
		args_copy['save_dir'] = str(args.save_dir)
		json.dump(args_copy, f, indent=2)

	if 'bt' in args.model:
		model = BarlowTwins(args)
	elif 'supervised' in args.model:
		model = ResNet18(args)
	else:
		raise NotImplementedError('Model not supported.')

	if args.mode == 'train':
		train(args, model)
	else:
		pass


if __name__ == '__main__':
	main()