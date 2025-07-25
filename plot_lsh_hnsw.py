import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to plot!")

filepath, _ = os.path.splitext(sys.argv[1])

qualities = pd.read_hdf(filepath + "_results.h5", key="qualities")
timings = pd.read_hdf(filepath + "_results.h5", key="timings")
ranks = pd.read_hdf(filepath + "_results.h5", key="ranks")

results = h5py.File(filepath + "_results.h5", 'r')

var_name = results.attrs['var_name']
print(f'var_name : {var_name}')

sns.lineplot(data=qualities, x=var_name, y='Quality', hue='algo')

plt.savefig(filepath + "_qualities.pdf")

plt.clf()

sns.boxplot(data=ranks, x=var_name, y='Rank', hue='algo')

plt.yscale('log', base=2)

plt.savefig(filepath + "_ranks.pdf")

plt.clf()

sns.boxplot(data=timings, x='event', y='dt', hue='algo')

plt.savefig(filepath + "_timings.pdf")
