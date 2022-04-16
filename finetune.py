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

from loading_utils import get_finetune_dataloaders

def prepare_model(dir_name):


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

	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

	trainset, testset = get_datasets(args)
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	criterion = nn.CrossEntropyLoss().to(device)

	model.train()
	start_time = time.time()
	start_epoch = 0
	
	for epoch in range(start_epoch, args.epochs):

		loss_total = 0

		for step, (batch, labels) in enumerate(train_loader, start=epoch*len(train_loader)):

			inputs = batch.cuda(non_blocking=True)
			targets = labels.cuda(non_blocking=True)

			adjust_learning_rate(args, optimizer, train_loader, step)
			optimizer.zero_grad()

			loss = criterion(model(inputs), targets)

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