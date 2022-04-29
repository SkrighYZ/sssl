import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

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
stm_num_replays_per_sample = np.zeros(len(samples), dtype=int)
stm_num_replays_per_class = np.zeros(num_classes, dtype=int)
stm_num_replays_per_clip = np.zeros(curr_clip+1, dtype=int)
for b in stm_batches:
	for i in b:
		stm_num_replays_per_sample[i] += 1
		stm_num_replays_per_class[class_labels[i]] += 1
		stm_num_replays_per_clip[clip_labels[i]] += 1
		stm_sample_dist.append(i)
		stm_class_dist.append(class_labels[i])
		stm_clip_dist.append(clip_labels[i])


ltm_sample_dist = []
ltm_class_dist = []
ltm_clip_dist = []
ltm_num_replays_per_sample = np.zeros(len(samples), dtype=int)
ltm_num_replays_per_class = np.zeros(num_classes, dtype=int)
ltm_num_replays_per_clip = np.zeros(curr_clip+1, dtype=int)
for b in ltm_batches:
	for i in b:
		ltm_num_replays_per_sample[i] += 1
		ltm_num_replays_per_class[class_labels[i]] += 1
		ltm_num_replays_per_clip[clip_labels[i]] += 1
		ltm_sample_dist.append(i)
		ltm_class_dist.append(class_labels[i])
		ltm_clip_dist.append(clip_labels[i])


num_replays_per_class = stm_num_replays_per_class + ltm_num_replays_per_class
num_replays_per_sample = stm_num_replays_per_sample + ltm_num_replays_per_sample
num_replays_per_clip = stm_num_replays_per_clip + ltm_num_replays_per_clip

# stm_num_replays_per_sample = [s for s in stm_num_replays_per_sample if s != 0]
# ltm_num_replays_per_sample = [s for s in ltm_num_replays_per_sample if s != 0]
# stm_num_replays_per_clip  = [s for s in stm_num_replays_per_clip if s != 0]
# ltm_num_replays_per_clip = [s for s in ltm_num_replays_per_clip if s != 0]

# num_replays_per_sample = [s for s in num_replays_per_sample if s != 0]
# num_replays_per_clip = [s for s in num_replays_per_clip if s != 0]


#bins = np.linspace(0, len(samples), 50)
bins = np.linspace(0, 50, 51)
#bins = np.linspace(0, curr_clip+1, 50)
_, _, patches = plt.hist([ltm_class_dist, stm_class_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
#plt.hist(stm_class_dist, bins, stacked=True, align="mid", alpha=0.5)

# for pp in patches:
#    x = (pp._x0 + pp._x1)/2-10
#    y = pp._y1 + 0.05
#    plt.text(x, y, int(pp._y1))

plt.ylabel('Num Replays')
plt.xlabel('Class Index')
plt.legend()
plt.show()