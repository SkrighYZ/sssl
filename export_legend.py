import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle

plt.style.use('seaborn')
FONT_SIZE = 16

def export_legend(save_dir, handles, labels, FONT_SIZE=14):
	# this function is used to save png files of the legend

	# vertical legend
	fig_legend = plt.figure(figsize=(1.7, 1.7))
	fig_legend.legend(handles, labels, loc='center', frameon=True, ncol=1, fontsize=FONT_SIZE)
	fig_legend.savefig(os.path.join(save_dir, 'legend2.png'),
					   bbox_inches="tight", format='png')

	# horizontal legend
	fig_legend = plt.figure(figsize=(10, 0.4))

	fig_legend.legend(handles, labels, loc='center', frameon=True, ncol=6, fontsize=FONT_SIZE)
	fig_legend.savefig(os.path.join(save_dir, 'legend.png'),
					   bbox_inches="tight", format='png')


fig, ax = plt.subplots()

groups = ['Span=500', 'Span=1000', 'Span=1500', 'Span=2000', 'Span=3000', 'IID']


for i in range(len(groups)):
	plt.plot([1, 2, 3], [1, 2, 3], label=groups[i])






handles, labels = ax.get_legend_handles_labels()
export_legend('../sssl-ws/plots', handles, labels)