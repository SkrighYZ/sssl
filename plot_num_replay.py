import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
import random

plt.style.use('seaborn')

save_dir = sys.argv[1]
num_classes = 51

samples = pickle.load(open('../sssl-ws/' + save_dir + '/samples.pkl', 'rb'))
stm_batches = pickle.load(open('../sssl-ws/' + save_dir + '/stm_batches.pkl', 'rb'))
ltm_batches = pickle.load(open('../sssl-ws/' + save_dir + '/ltm_batches.pkl', 'rb'))


clip_labels = []
curr_clip = -1
for s in samples:
	if s[3] == 0:
		curr_clip += 1
	clip_labels.append(curr_clip)
class_labels = [s[0] for s in samples]


stm_sample_dist = []
stm_class_dist = []
stm_clip_dist = []

ltm_sample_dist = []
ltm_class_dist = []
ltm_clip_dist = []

for b_i in range(len(stm_batches)):

	for i in stm_batches[b_i]:
		stm_sample_dist.append(i)
		stm_class_dist.append(class_labels[i])
		stm_clip_dist.append(clip_labels[i])

	for i in ltm_batches[b_i]:
		ltm_sample_dist.append(i)
		ltm_class_dist.append(class_labels[i])
		ltm_clip_dist.append(clip_labels[i])

unused_num = len(samples)-len(set(stm_sample_dist+ltm_sample_dist))
print('Percent of examples not used:', unused_num / len(samples))

bins = np.linspace(0, 50, 51)
_, _, patches = plt.hist([ltm_class_dist, stm_class_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
plt.ylabel('Num Replays')
plt.xlabel('Class Index')
plt.legend()
plt.show()

bins = np.linspace(0, curr_clip+1, 100)
_, _, patches = plt.hist([ltm_clip_dist, stm_clip_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
plt.ylabel('Num Replays')
plt.xlabel('Clip Index')
plt.legend()
plt.show()

bins = np.linspace(0, len(samples), 100)
_, _, patches = plt.hist([ltm_sample_dist, stm_sample_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
plt.ylabel('Num Replays')
plt.xlabel('Sample Index')
plt.legend()
plt.show()

sample_dist = stm_sample_dist + ltm_sample_dist
sample_occurance = {}
for ex in sample_dist:
	if ex in sample_occurance:
		sample_occurance[ex] += 1
	else:
		sample_occurance[ex] = 1
sample_counts = list(sample_occurance.values())
bins = np.linspace(0, max(sample_counts), 100)
plt.hist(sample_counts, bins)
plt.show()

