import pickle
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import random
import seaborn as sns
from statistics import mean
from matplotlib.ticker import StrMethodFormatter

plt.style.use('seaborn')
sns.set_style('darkgrid', {'legend.frameon':True})

cat = sys.argv[1]  # clip | class
mem = sys.argv[2]  # LTM+STM | LTM | STM


def smooth(y, box_pts=40):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth



num_classes = 51


comparison = ['r_bt_span500', 'r_bt_span1000', 'r_bt_span1500', 'r_bt_span2000', 'r_bt_span3000']
labels = ['Span=500', 'Span=1000', 'Span=1500', 'Span=2000', 'Span=3000', 'IID']

plt.rcParams["figure.figsize"] = (4, 4)
fig, ax = plt.subplots()

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
			plt.plot(smooth(ltm_class_per_batch), label=labels[c_i])
		elif mem == 'STM':
			plt.plot(smooth(stm_class_per_batch), label=labels[c_i])
		else:
			plt.plot(smooth(total_class_per_batch), label=labels[c_i])
	elif cat == 'clip':
		if mem == 'LTM':
			plt.plot(smooth(ltm_clip_per_batch), label=labels[c_i])
		elif mem == 'STM':
			plt.plot(smooth(stm_clip_per_batch), label=labels[c_i])
		else:
			plt.plot(smooth(total_clip_per_batch), label=labels[c_i])



if cat == 'class':
	if mem == 'LTM+STM':
		plt.plot(smooth(iid_class_per_batch), label='IID')
	plt.ylabel('Num. Classes', fontweight='bold', fontsize=22)


elif cat == 'clip':
	if mem == 'LTM+STM':
		plt.plot(smooth(iid_clip_per_batch), label='IID')
	plt.ylabel('Num. Clips', fontweight='bold', fontsize=22)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xlabel('Step', fontweight='bold', fontsize=22)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
#plt.legend(fontsize=16, frameon=1, framealpha=1)
	
plt.tight_layout()
plt.show()
fig.savefig(os.path.join('../sssl-ws/plots', 'span_'+mem+'_'+cat+'.png'),
					bbox_inches="tight", format='png')

