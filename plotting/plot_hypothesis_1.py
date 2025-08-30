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
    'rand sampling' : '#2ca02c'
}

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to plot!")

filepath, _ = os.path.splitext(sys.argv[1])
filepath = filepath[:-8]

qualities = pd.read_hdf(filepath + "_results.h5", key="qualities")
timings = pd.read_hdf(filepath + "_results.h5", key="timings")
ranks = pd.read_hdf(filepath + "_results.h5", key="ranks")
recalls = pd.read_hdf(filepath + "_results.h5", key="recalls")

results = h5py.File(filepath + "_results.h5", 'r')

var_name = results.attrs['var_name']
var_values = results.attrs['var_values'][()]

plot_path = "plots/" + filepath[5:]

fig, ax = plt.subplots()

sns.lineplot(data=qualities, x='instance', y='Quality', hue='algo', palette=algo_to_color, legend=False)

ax.set(xlabel=var_name)
pos = range(len(var_values))
labels = [f'$2^{{{int(math.log2(v))}}}$' for v in var_values]
plt.xticks(pos, labels)

plt.savefig(plot_path + "_qualities.pdf")

plt.clf()

fig, ax = plt.subplots()

sns.boxplot(data=ranks, x='algo', y='Rank', hue='instance', legend=False, fliersize=0)

ax.set(xlabel='Instances per Algorithm')
ax.set_xticklabels([])

# Hacky way of properly coloring the groups. Found nothing for this when searching docs and forums.
colors = list(algo_to_color.values())
for patch_idx, patch in enumerate(ax.patches):
    patch.set_facecolor(colors[patch_idx % len(colors)])

plt.yscale('log', base=2)

plt.savefig(plot_path + "_ranks.pdf")

plt.clf()

fig, ax = plt.subplots()

sns.lineplot(data=timings[(timings['event'] == 'creation') & (timings['algo'] != 'rand sampling')], x='instance', y='dt', hue='algo', palette=algo_to_color, legend=False, ax=ax)

ax.set_ylabel('Time in Seconds')
ax.set(xlabel=var_name)
pos = range(len(var_values))
labels = [f'$2^{{{int(math.log2(v))}}}$' for v in var_values]
plt.xticks(pos, labels)

plt.yscale('log', base=2)

plt.savefig(plot_path + "_timings_creation.pdf")

plt.clf()

fig, ax = plt.subplots()

sns.lineplot(data=timings[(timings['event'] == 'query') & (timings['algo'] != 'rand sampling')], x='instance', y='dt', hue='algo', palette=algo_to_color, legend=False, ax=ax)

ax.set_ylabel('Time in Seconds')
ax.set(xlabel=var_name)
pos = range(len(var_values))
labels = [f'$2^{{{int(math.log2(v))}}}$' for v in var_values]
plt.xticks(pos, labels)

plt.yscale('log', base=2)

plt.savefig(plot_path + "_timings_query.pdf")

plt.clf()

fig, ax = plt.subplots()

sns.boxplot(data=recalls, x='algo', y='Recall', hue='instance', legend=False, ax=ax, showfliers=False)

ax.set(xlabel='Instances per Algorithm')
ax.set_xticklabels([])

# Hacky way of properly coloring the groups. Found nothing for this when searching docs and forums.
colors = list(algo_to_color.values())
for patch_idx, patch in enumerate(ax.patches):
    patch.set_facecolor(colors[patch_idx % len(colors)])

plt.savefig(plot_path + "_recalls.pdf")

plt.clf()

sns.lineplot(data=recalls, x='k', y='Recall', hue='algo', palette=algo_to_color, legend=False)

plt.savefig(plot_path + "_recalls@.pdf")
