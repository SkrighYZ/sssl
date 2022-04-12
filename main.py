import os
import argparse
import json
import time
import numpy as np
import math
import random
import sys
from pathlib import Path

from torch import optim, nn
import torch

from models import BarlowTwins

from loading_utils import get_stream_data_loaders


def train(args, model, device='cuda:0'):

	model.to(device)

	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

	# automatically resume from checkpoint if it exists
	if (args.save_dir / 'checkpoint.pth').is_file():
		ckpt = torch.load(args.save_dir / 'checkpoint.pth', map_location='cpu')
		start_epoch = ckpt['epoch']
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
	else:
		start_epoch = 0

	dataset, train_loader, replay_loader, replay_sampler = get_stream_data_loaders(args.images_dir, args.dataset, args.order, args.batch_size, num_workers=args.num_workers)

	start_time = time.time()
	for epoch in range(start_epoch, args.epochs):

		dataset.shuffle()
		replay_sampler.rehearsal_ixs = list(range(args.batch_size-1))
		replay_iter = iter(replay_loader)

		for step, ((y1, y2), _) in enumerate(train_loader, start=epoch*len(train_loader)):
			
			if step < args.batch_size-1:
				continue

			# Update sliding window buffer
			if step < args.buffer_size:
				replay_sampler.rehearsal_ixs += [step]
			else:
				replay_sampler.rehearsal_ixs = replay_sampler.rehearsal_ixs[1:] + [step]

			if step+1 % args.batch_size != 0:
				continue

			y1 = y1.cuda(non_blocking=True)
			y2 = y2.cuda(non_blocking=True)

			(replay_y1, replay_y2), _ = next(replay_iter)
			replay_y1 = replay_y1.cuda(non_blocking=True)
			replay_y2 = replay_y2.cuda(non_blocking=True)

			y1_inputs = torch.cat([y1, replay_y1], dim=0)
			y2_inputs = torch.cat([y2, replay_y2], dim=0)

			optimizer.zero_grad()
			loss = model(y1_inputs, y2_inputs)
			loss.backward()
			optimizer.step()

			if step % args.print_freq == 0:
				stats = dict(epoch=epoch,
							step=step,
							total=len(train_loader), 
							lr=optimizer.param_groups[0]['lr'],
							loss=loss.item(),
							time=int(time.time() - start_time))
				print(json.dumps(stats))

			if step % args.save_freq == 0:
				state = dict(epoch=epoch, step=step, model=model.state_dict(), optimizer=optimizer.state_dict())
				torch.save(state, args.save_dir / 'checkpoint.pth')

		state = dict(epoch=epoch, step=step, model=model.state_dict(), optimizer=optimizer.state_dict())
		torch.save(state, args.save_dir / 'checkpoint.pth')

	torch.save(model.backbone.state_dict(), args.save_dir / 'resnet50.pth')

	return model



def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default='stream51', choices=['stream51'])
	parser.add_argument('--order', type=str, default='instance', choices=['iid', 'instance'])
	parser.add_argument('--model', type=str, default='sliding_bt',
						choices=['sliding_bt', 'sliding_simclr', 'reservoir_bt', 'cluster_bt', 'hnm_simclr'])

	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--buffer_size', type=int, default=63)

	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=0.2)
	parser.add_argument('--weight_decay', type=float, default=1e-6)
	parser.add_argument('--momentum', default=0.9, type=float)

	parser.add_argument('--seed', type=int, default=10)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--print-freq', default=1000, type=int, metavar='N', help='print frequency')
	parser.add_argument('--save-freq', default=10000, type=int, metavar='N', help='save frequency')

	parser.add_argument('--projector', default='2048-2048', type=str, metavar='MLP', help='projector MLP')
	parser.add_argument('--lambd', default=0.0051, type=float, metavar='L', help='weight on off-diagonal terms')

	parser.add_argument('--images_dir', type=Path, metavar='DIR', default='../data/Stream-51')
	parser.add_argument('--save_dir', type=Path, metavar='DIR')

	args = parser.parse_args()
	print(args)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if args.model == 'sliding_bt':
		torch.manual_seed(args.seed)
		model = BarlowTwins(args)
	else:
		raise NotImplementedError('Model not supported.')

	# perform streaming classification
	train(args, model)


if __name__ == '__main__':
	main()