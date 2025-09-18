#!/usr/bin/env python3
import pandas as pd
import math
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import sys

algo_to_color = {
    'lsh' : '#d62728',
    'hnsw' : '#9467bd',
    'kd@0.2' : '#7cbce9',
    'kd@0.4' : '#51a6e1',
    'kd@0.6' : '#2590da',
    'kd@0.8' : '#1e73ae',
    'kd@1.0' : '#165683',
    'bt@0.2' : '#ffae66',
    'bt@0.4' : '#ff9333',
    'bt@0.6' : '#ff7800',
    'bt@0.8' : '#cc6000',
    'bt@1.0' : '#994800',
    'rand sampling' : '#2ca02c',
    'bruteforce' : '#DFF954'
}

filepath =  'data/hypothesis_9_further'

recalls = pd.read_hdf(filepath + "_results.h5", key="recalls")

results = h5py.File(filepath + "_results.h5", 'r')

var_name = results.attrs['var_name']
var_values = results.attrs['var_values'][()]

plot_path = "plots/" + filepath[5:]

fig, ax = plt.subplots()

sns.lineplot(data=recalls[recalls['algo'] == 'lsh'], x='k', y='Recall', hue='sample_rate', legend=False)

plt.ylim(0.0, 1.0)

plt.savefig(plot_path + "_recalls@.pdf")

plt.clf()

lids = recalls.groupby(['instance', 'sample_rate']).agg(MeanRecall=("Recall", "mean")).reset_index()
lids = lids.rename(columns = {'instance':'LID', "sample_rate":"Sample Rate"})
lids['LID'] = lids['LID'].replace(5, 60).replace(0,5).replace(1,20).replace(2,30).replace(3,40) \
        .replace(4,50).replace(6,70).replace(7,80).replace(8, 90).replace(9, 100)
lids = lids.pivot(index='LID', columns='Sample Rate', values='MeanRecall')

sns.heatmap(lids, cmap='Reds')
plt.savefig(plot_path + "_recall_heatmap.pdf")
