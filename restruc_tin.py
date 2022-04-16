import glob
import os
from shutil import copy
import config

old_folder = '../data/tiny-imagenet-200/val/'
new_folder = '../data/tiny-imagenet-200/val_restruc/'
if not os.path.exists(new_folder):
	os.mkdir(new_folder)

val_dict = {}
with open(old_folder + 'val_annotations.txt', 'r') as f:
	for line in f.readlines():
		split_line = line.split('\t')
		val_dict[split_line[0]] = split_line[1]
		
paths = glob.glob(old_folder + 'images/*')
for path in paths:
	file = path.split('/')[-1]
	folder = val_dict[file]
	if not os.path.exists(new_folder + str(folder)):
		os.mkdir(new_folder + str(folder))
		
for path in paths:
	file = path.split('/')[-1]
	folder = val_dict[file]
	dest = new_folder + str(folder) + '/' + str(file)
	copy(path, dest)