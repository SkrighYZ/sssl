import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
from statistics import mean

plt.style.use('seaborn')

save_dir = sys.argv[1]
num_classes = 51

samples = pickle.load(open('../sssl-ws/' + save_dir + '/samples.pkl', 'rb'))
stm_batches = pickle.load(open('../sssl-ws/' + save_dir + '/stm_batches.pkl', 'rb'))
ltm_batches = pickle.load(open('../sssl-ws/' + save_dir + '/ltm_batches.pkl', 'rb'))

print(ltm_batches[-5:])

clip_labels = []
curr_clip = -1
for s in samples:
	if s[3] == 0:
		curr_clip += 1
	clip_labels.append(curr_clip)
class_labels = [s[0] for s in samples]

num_samples = (stm_batches.shape[0]-1) * (stm_batches.shape[1]+ltm_batches.shape[1])
iid_batches = list(range(num_samples))
random.shuffle(iid_batches)
iid_batches = np.array(iid_batches, dtype=int).reshape(
		(stm_batches.shape[0]-1, stm_batches.shape[1]+ltm_batches.shape[1]))


stm_class_per_batch = []
stm_clip_per_batch = []

ltm_class_per_batch = []
ltm_clip_per_batch = []

total_class_per_batch = []
total_clip_per_batch = []

iid_class_per_batch = []
iid_clip_per_batch = []

for b_i in range(len(stm_batches)):

	stm_class_batch = set()
	stm_clip_batch = set()
	for i in stm_batches[b_i]:
		stm_class_batch.add(class_labels[i])
		stm_clip_batch.add(clip_labels[i])

	stm_class_per_batch.append(len(stm_class_batch))
	stm_clip_per_batch.append(len(stm_clip_batch))

	ltm_class_batch = set()
	ltm_clip_batch = set()
	for i in ltm_batches[b_i]:
		ltm_class_batch.add(class_labels[i])
		ltm_clip_batch.add(clip_labels[i])

	ltm_class_per_batch.append(len(ltm_class_batch))
	ltm_clip_per_batch.append(len(ltm_clip_batch))

	total_class_batch = stm_class_batch.union(ltm_class_batch)
	total_clip_batch = stm_clip_batch.union(ltm_clip_batch)
	total_class_per_batch.append(len(total_class_batch))
	total_clip_per_batch.append(len(total_clip_batch))

	if b_i == iid_batches.shape[0]: break

	iid_class_batch = set()
	iid_clip_batch = set()
	for i in iid_batches[b_i]:
		iid_class_batch.add(class_labels[i])
		iid_clip_batch.add(clip_labels[i])
	iid_class_per_batch.append(len(iid_class_batch))
	iid_clip_per_batch.append(len(iid_clip_batch))



print('------------NUM CLASSES------------')
print('Mean LTM:', mean(ltm_class_per_batch[200:]))
print('Mean STM:', mean(stm_class_per_batch[200:]))
print('Mean LTM+STM:', mean(total_class_per_batch[200:]))
print('Mean IID:', mean(iid_class_per_batch[200:]))

plt.plot(ltm_class_per_batch, label='LTM')
plt.plot(stm_class_per_batch, label='STM')
plt.plot(total_class_per_batch, label='LTM+STM')
plt.plot(iid_class_per_batch, label='IID')
upper_bound = [51] * len(ltm_class_per_batch)
plt.plot(upper_bound, ls='--', label='Upper Bound')
plt.ylim((0, 55))
plt.ylabel('Num Classes Within Batch')
plt.xlabel('Step')
plt.legend()
plt.gcf().set_size_inches(5, 5, forward=True)
plt.show()


print('------------NUM CLIPS------------')
print('Mean LTM:', mean(ltm_clip_per_batch[200:]))
print('Mean STM:', mean(stm_clip_per_batch[200:]))
print('Mean LTM+STM:', mean(total_clip_per_batch[200:]))
print('Mean IID:', mean(iid_clip_per_batch[200:]))

plt.plot(ltm_clip_per_batch, label='LTM')
plt.plot(stm_clip_per_batch, label='STM')
plt.plot(total_clip_per_batch, label='LTM+STM')
plt.plot(iid_clip_per_batch, label='IID')
upper_bound = [128] * len(ltm_clip_per_batch)
plt.plot(upper_bound, ls='--', label='Upper Bound')
plt.ylim((0, 130))
plt.ylabel('Num Clips Within Batch')
plt.xlabel('Step')
plt.legend()
plt.gcf().set_size_inches(5, 5, forward=True)
plt.show()




