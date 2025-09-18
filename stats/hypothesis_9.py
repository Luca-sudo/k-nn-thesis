#!/usr/bin/env python3
import pandas as pd
from tabulate import tabulate

filepath = "data/hypothesis_9_results.h5"

recalls = pd.read_hdf(filepath, key="recalls")
timings = pd.read_hdf(filepath, key="timings")

r = recalls[recalls['algo'].isin(['lsh','hnsw'])]
q_times = timings[(timings['algo'].isin(['lsh', 'hnsw'])) & (timings['event'] == 'query')]
c_times = timings[(timings['algo'].isin(['lsh', 'hnsw'])) & (timings['event'] == 'creation')]

recall_at_k = r.groupby(['algo', 'k']).agg(MeanRecall=("Recall", "mean")).reset_index()
delta_at_k = recall_at_k.pivot(index='k', columns='algo', values='MeanRecall')
delta_at_k['delta'] = delta_at_k['hnsw'] - delta_at_k['lsh']
delta_at_k = delta_at_k.T.reset_index(drop=True)
delta_at_k.index = ['hnsw', 'lsh', 'delta']

print('\nDifferences in mean recall between LSH and HNSW, grouped by each k')
print(tabulate(delta_at_k, headers='keys'))

recall_at_instance = r.groupby(['algo', 'instance', 'k']).agg(MeanRecall=("Recall", "mean")).reset_index()
#print(recall_at_instance)

# Step 1: Filter data for instances 0 and 9, then merge
inst0 = recall_at_instance[recall_at_instance['instance'] == 0][['k', 'algo', 'MeanRecall']].rename(columns={'MeanRecall': 'recall_instance_0'})
inst9 = recall_at_instance[recall_at_instance['instance'] == 9][['k', 'algo', 'MeanRecall']].rename(columns={'MeanRecall': 'recall_instance_9'})

# Step 2: Merge on k and algo
merged = pd.merge(inst0, inst9, on=['k', 'algo'])

# Step 3: Calculate fraction (instance_0 / instance_9)
merged['degradation'] = merged['recall_instance_9'] / merged['recall_instance_0']

# Step 4: Pivot to get algorithms as columns
result = merged.pivot(index='k', columns='algo')

# Step 5: Flatten multi-index columns
result.columns = ['_'.join(col).strip() for col in result.columns.values]
result = result.reset_index()
result = result[['degradation_hnsw', 'degradation_lsh']].T
result = result.rename(lambda x: str(int(x) * 10 + 10), axis='columns')
print('\nFor fixed values of k, calculate the recall degradation from first to last instance, computed as the fraction of achieved recalls.')
print(tabulate(result, headers='keys'))

creation_delta = c_times.pivot(index='algo', columns='instance', values='dt')
creation_delta.loc['fractional difference'] = creation_delta.loc['hnsw'] / creation_delta.loc['lsh']
print('\nCalculate the absolute time to create index for each instance, and compare the fraction between HNSW & LSH')
print(creation_delta)

query_delta = q_times.groupby(['algo', 'instance']).agg(MeanDelta=("dt", "mean")).reset_index().pivot(index='algo', columns='instance', values='MeanDelta')
query_delta.loc['fractional difference'] = query_delta.loc['hnsw'] / query_delta.loc['lsh']
print('\nCalculate the mean time for queries over each instance, and compare fractionally.')
print(query_delta)

print(r[(r['algo'] == 'hnsw') & r['instance'] == 0].groupby(['instance'])[['instance', 'Recall']].mean())
