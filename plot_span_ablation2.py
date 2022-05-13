import matplotlib.pyplot as plt 
import os
import statistics
import numpy as np
import sys
import seaborn as sns

stm_span = np.array([500, 1000, 1500, 2000, 3000])
bt_acc = [10.59, 11.44, 15.22, 15.35, 16.32]
simclr_acc = [19.05, 19.56, 18.88, 19.19, 19.13]

plt.style.use('seaborn')

colors = sns.color_palette()
xlabel = 'Log STM Size'
ylabels = ['Num. Classes', 'Num. Clips', 'Example Used [%]', 'Accuracy [%]']
save_names = ['class', 'clip', 'ss', 'acc']

plt.rcParams["figure.figsize"] = (6, 4)
fig, ax = plt.subplots()

plt.plot(stm_span, bt_acc, marker='o', label='BarlowTwins')
plt.plot(stm_span, simclr_acc, marker='^', label='SimCLR')
plt.xlabel('STM Span', fontweight='bold', fontsize=22)
plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=22)
plt.xticks(stm_span, stm_span, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join('../sssl-ws/plots', 'span_acc.png'),
					bbox_inches="tight", format='png')