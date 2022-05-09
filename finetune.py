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
import torchvision

from models import BarlowTwins, SimCLR, ResNet18
from loading_utils import get_finetune_data_loaders


def prepare_model(args):
	model = torchvision.models.resnet18(pretrained=False)
	model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
	model.maxpool = nn.Sequential()

	if 'random' in str(args.ckpt_dir):
		pass
	elif 'bt' in str(args.ckpt_dir) or 'simclr' in str(args.ckpt_dir):
		model.fc = nn.Identity()
		ckpt = torch.load(args.ckpt_dir / 'resnet18.pth', map_location='cpu')
		model.load_state_dict(ckpt)
	elif 'sup' in str(args.ckpt_dir):
		model.fc = nn.Linear(512, args.pretrained_num_classes)
		ckpt = torch.load(args.ckpt_dir / 'resnet18.pth', map_location='cpu')
		model.load_state_dict(ckpt)
	else:
		raise NotImplementedError('Model not supported.')

	model.fc = nn.Linear(512, args.num_classes)

	params_to_learn = []
	for name, param in model.named_parameters():
		if name.startswith('fc'):
			params_to_learn.append(param)
		else:
			param.requires_grad = False

	return model, params_to_learn


def train(args, model, params_to_learn, device='cuda:0'):

	model.to(device)

	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = optim.SGD(params_to_learn, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epochs, gamma=args.lr_decay)

	train_loader, test_loader = get_finetune_data_loaders(args)

	start_time = time.time()
	start_epoch = 0
	training_logs = {'loss': [], 'acc': []}
	
	for epoch in range(start_epoch, args.epochs):

		loss_total = 0
		model.train()

		for step, (data, labels) in enumerate(train_loader, start=epoch*len(train_loader)):

			inputs = data.cuda(non_blocking=True)
			targets = labels.cuda(non_blocking=True)

			optimizer.zero_grad()
			loss = criterion(model(inputs), targets)
			loss.backward()
			optimizer.step()

			loss_total += loss.item()

		scheduler.step()

		stats = dict(epoch=epoch,
				lr=optimizer.param_groups[0]['lr'],
				avg_loss=loss_total/len(train_loader),
				time=int(time.time() - start_time))
		training_logs['loss'].append(loss_total/len(train_loader))
		loss_total = 0
		print(json.dumps(stats))

		if (epoch+1) % args.eval_freq == 0:
			acc = eval(args, model, test_loader)
			training_logs['acc'].append(acc)

		with open(args.ckpt_dir / 'finetune_logs.json', 'w') as f:
			json.dump(training_logs, f, indent=4)

	torch.save(model.state_dict(), args.ckpt_dir / ('linear_'+args.dataset+'.pth'))

	return model


def eval(args, model, test_loader):
	model.eval()
	correct = 0
	with torch.no_grad():
		for data, labels in test_loader:
			inputs = data.cuda(non_blocking=True)
			targets = labels.cuda(non_blocking=True)
			preds = model(inputs)
			correct += (preds.argmax(1) == targets).type(torch.float).sum().item()

	acc = correct / len(test_loader.dataset)
	print('Testing Accuracy: {}'.format(acc))

	return acc



def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default='tinyimagenet', choices=['tinyimagenet'])
	parser.add_argument('--pretrained_num_classes', type=int, default=51)
	parser.add_argument('--num_classes', type=int, default=200)

	parser.add_argument('--batch_size', type=int, default=256)

	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--learning_rate', type=float, default=0.1)
	parser.add_argument('--lr_decay', type=float, default=0.1)
	parser.add_argument('--decay_epochs', nargs='+', type=int, default=[20, 40], help='learning rate decay epochs')
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--momentum', default=0.9, type=float)

	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--eval_freq', default=10, type=int, metavar='N', help='eval frequency')

	parser.add_argument('--ckpt_dir', type=Path, metavar='DIR')
	parser.add_argument('--images_dir', type=Path, metavar='DIR', default='../data/tiny-imagenet-200')

	args = parser.parse_args()
	print(args)

	if not os.path.exists(args.ckpt_dir):
		raise Exception('Incorrect directory.')

	model, params_to_learn = prepare_model(args)

	train(args, model, params_to_learn)


if __name__ == '__main__':
	main()