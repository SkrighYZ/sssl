import matplotlib.pyplot as plt 
import os
import statistics
import numpy as np
import sys
import seaborn as sns

stm_size = np.array([0, 2, 8, 32, 128, 256])
avg_class_per_batch = [47.12, 45.52, 45.95, 46, 40.6, 25]
avg_clip_per_batch = [109.35, 111.83, 110.5, 105.87, 81.44, 33.3]
percent_ex_used = [1.18, 1.25, 1.61, 2.89, 8.14, 15.16]
accuracy = [15.71, 15.69, 15.82, 15.01, 15.22, 9.08]
data = [avg_class_per_batch, avg_clip_per_batch, percent_ex_used, accuracy]

plt.style.use('seaborn')

colors = sns.color_palette()
xlabel = 'Log STM Size'
ylabels = ['Avg. Num. Classes', 'Avg. Num. Clips', 'Example Used [%]', 'Accuracy [%]']
save_names = ['class', 'clip', 'ss', 'acc']

for i in range(4):
	plt.rcParams["figure.figsize"] = (4, 4)
	fig, ax = plt.subplots()
	plt.plot(np.log2(stm_size+1), data[i], marker='o', color=colors[i])
	plt.yticks(fontsize=16)
	plt.xticks(np.log2(stm_size+1), stm_size, fontsize=16)
	plt.xlabel(xlabel, fontweight='bold', fontsize=22)
	plt.ylabel(ylabels[i], fontweight='bold', fontsize=22)
	plt.tight_layout()
	plt.show()

	fig.savefig(os.path.join('../sssl-ws/plots', 'stm_size_'+save_names[i]+'.png'),
					bbox_inches="tight", format='png')