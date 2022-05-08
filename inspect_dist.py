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

stm_class_per_batch = []
stm_clip_per_batch = []
for b in stm_batches:
	class_batch = set()
	clip_batch = set()
	for i in b:
		stm_sample_dist.append(i)
		stm_class_dist.append(class_labels[i])
		stm_clip_dist.append(clip_labels[i])

		class_batch.add(class_labels[i])
		clip_batch.add(clip_labels[i])

	stm_class_per_batch.append(len(class_batch))
	stm_clip_per_batch.append(len(clip_batch))

ltm_sample_dist = []
ltm_class_dist = []
ltm_clip_dist = []
ltm_class_per_batch = []
ltm_clip_per_batch = []
for b in ltm_batches:
	class_batch = set()
	clip_batch = set()
	for i in b:
		ltm_sample_dist.append(i)
		ltm_class_dist.append(class_labels[i])
		ltm_clip_dist.append(clip_labels[i])

		class_batch.add(class_labels[i])
		clip_batch.add(clip_labels[i])

	ltm_class_per_batch.append(len(class_batch))
	ltm_clip_per_batch.append(len(clip_batch))


# bins = np.linspace(0, 50, 51)
# _, _, patches = plt.hist([ltm_class_dist, stm_class_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
# plt.ylabel('Num Replays')
# plt.xlabel('Class Index')
# plt.legend()
# plt.show()

# bins = np.linspace(0, curr_clip+1, 50)
# _, _, patches = plt.hist([ltm_clip_dist, stm_clip_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
# plt.ylabel('Num Replays')
# plt.xlabel('Clip Index')
# plt.legend()
# plt.show()

# bins = np.linspace(0, len(samples), 50)
# _, _, patches = plt.hist([ltm_sample_dist, stm_sample_dist], bins, label=['LTM', 'STM'], stacked=True, align="mid")
# plt.ylabel('Num Replays')
# plt.xlabel('Sample Index')
# plt.legend()
# plt.show()

plt.plot(ltm_class_per_batch, label='LTM')
plt.plot(stm_class_per_batch, label='STM')
optimal = [51] * len(ltm_class_per_batch)
plt.plot(optimal, ls='--', label='optimal')
plt.ylim((0, 55))
plt.ylabel('Num Classes Within Batch')
plt.xlabel('Step')
plt.legend()
plt.show()

plt.plot(ltm_clip_per_batch, label='LTM')
plt.plot(stm_clip_per_batch, label='STM')
optimal = [64] * len(ltm_clip_per_batch)
plt.plot(optimal, ls='--', label='optimal')
plt.ylim((0, 70))
plt.ylabel('Num Clips Within Batch')
plt.xlabel('Step')
plt.legend()
plt.show()




