import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
from statistics import mean

plt.style.use('seaborn')

cat = sys.argv[1]  # clip | class
mem = sys.argv[2]  # LTM+STM | LTM | STM


def smooth(y, box_pts=20):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth



num_classes = 51

comparison = ['r_bt_s0_b128_sb0', 'r_bt_s2_b128_sb1', 'r_bt_s8_b128_sb4', 'r_bt_s32_b128_sb16', 'r_bt_s128_b128_sb64', 'r_bt_s256_b128_sb128']
#comparison = ['r_bt_span500', 'r_bt_span1000', 'r_bt_span1500', 'r_bt_span2000', 'r_bt_span3000', 'r_bt_span4000']
#comparison = ['r_bt_s128_b128_sb64', 'rb_bt_s128_b128_sb64', 'r_m_bt_s128_b128_sb64']
method = ['naive', 'boundary', 'min-replay']

for c_i, save_dir in enumerate(comparison):
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

	if cat == 'class':
		if mem == 'LTM':
			plt.plot(smooth(ltm_class_per_batch), label=save_dir)
		elif mem == 'STM':
			plt.plot(smooth(stm_class_per_batch), label=save_dir)
		else:
			plt.plot(smooth(total_class_per_batch), label=save_dir)
	elif cat == 'clip':
		if mem == 'LTM':
			plt.plot(smooth(ltm_clip_per_batch), label=save_dir)
		elif mem == 'STM':
			plt.plot(smooth(stm_clip_per_batch), label=save_dir)
		else:
			plt.plot(smooth(total_clip_per_batch), label=save_dir)



if cat == 'class':
	if mem == 'LTM+STM':
		plt.plot(smooth(iid_class_per_batch), label='IID')
	plt.ylim((0, 55))
	plt.ylabel('Num Classes Within Batch', fontsize=15)
	plt.xlabel('Step', fontsize=15)
	plt.legend(fontsize=12)
	plt.gcf().set_size_inches(5, 5, forward=True)


elif cat == 'clip':
	if mem == 'LTM+STM':
		plt.plot(smooth(iid_clip_per_batch), label='IID')
	plt.ylim((0, 130))
	plt.ylabel('Num Clips Within Batch', fontsize=15)
	plt.xlabel('Step', fontsize=15)
	plt.legend(fontsize=12)
	plt.gcf().set_size_inches(5, 5, forward=True)
	

plt.title(mem, fontsize=18)
plt.show()


