import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

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



stm_class_dist = []
stm_clip_dist = []
stm_num_replays_per_sample = np.zeros(len(samples), dtype=int)
for b in stm_batches:
	for i in b:
		stm_class_dist.append(class_labels[i])
		stm_clip_dist.append(clip_labels[i])
		stm_num_replays_per_sample[i] += 1


ltm_class_dist = []
ltm_clip_dist = []
ltm_num_replays_per_sample = np.zeros(len(samples), dtype=int)
for b in ltm_batches:
	for i in b:
		ltm_class_dist.append(class_labels[i])
		ltm_clip_dist.append(clip_labels[i])
		ltm_num_replays_per_sample[i] += 1



stm_num_replays_per_sample = [s for s in stm_num_replays_per_sample if s != 0]
ltm_num_replays_per_sample = [s for s in ltm_num_replays_per_sample if s != 0]



plt.hist(stm_num_replays_per_sample)
plt.show()