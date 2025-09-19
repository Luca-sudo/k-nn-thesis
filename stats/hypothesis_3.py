#!/usr/bin/env python3
import pandas as pd

filepath = "data/hypothesis_3_results.h5"

recalls = pd.read_hdf(filepath, key="recalls")
timings = pd.read_hdf(filepath, key="timings")

r = recalls[recalls['algo'].isin(['lsh','hnsw'])]
q_times = timings[(timings['algo'].isin(['lsh', 'hnsw'])) & (timings['event'] == 'query')]
c_times = timings[(timings['algo'].isin(['lsh', 'hnsw'])) & (timings['event'] == 'creation')]

print(r.groupby(['algo', 'instance']).agg(MeanRecall=("Recall", "mean")).reset_index().pivot(index='algo', columns='instance', values='MeanRecall'))

print(r[r['algo'] == 'lsh'].groupby(['k', 'instance']).agg(MeanRecall=("Recall", "mean")).reset_index().pivot(index='instance', columns='k', values='MeanRecall'))

recall_at_k = r.groupby(['algo', 'k']).agg(MeanRecall=("Recall", "mean")).reset_index()
delta_at_k = recall_at_k.pivot(index='k', columns='algo', values='MeanRecall')
delta_at_k['delta'] = delta_at_k['hnsw'] - delta_at_k['lsh']
delta_at_k = delta_at_k.T.reset_index(drop=True)
delta_at_k.index = ['hnsw', 'lsh', 'delta']

filepath = "data/hypothesis_3_2_results.h5"

recalls = pd.read_hdf(filepath, key="recalls")
timings = pd.read_hdf(filepath, key="timings")

r = recalls[recalls['algo'].isin(['lsh','hnsw'])]

print(r[r['algo'] == 'lsh'].groupby(['k', 'instance']).agg(MeanRecall=("Recall", "mean")).reset_index().pivot(index='instance', columns='k', values='MeanRecall'))
