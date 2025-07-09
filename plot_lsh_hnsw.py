import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to plot!")

filepath, _ = os.path.splitext(sys.argv[1])

qualities = pd.read_csv(filepath + "_qualities.csv")
timings = pd.read_csv(filepath + "_timings.csv")
ranks = pd.read_csv(filepath + "_ranks.csv")

sns.lineplot(data=qualities, x='instance', y='value', hue='algo')

plt.savefig(filepath + "_qualities.pdf")

plt.clf()

sns.boxplot(data=ranks, x='instance', y='value', hue='algo')

plt.yscale('log', base=2)

plt.savefig(filepath + "_ranks.pdf")

plt.clf()

sns.boxplot(data=timings, x='event', y='dt', hue='algo')

plt.savefig(filepath + "_timings.pdf")
