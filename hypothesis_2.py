#!/usr/bin/env python3

# Hypothesis 2: HNSW quality remains steady while LSH quality improves with increased spread (due to higher cosine similarity).

import time
import numpy as np
import h5py
import faiss

n_sites = 100000
n_dims = 100
k = 5
spreads = [2.0, 2.0 ** 1, 2.0 ** 2, 2.0 ** 3, 2.0 ** 4, 2.0 ** 5, 2.0 ** 6, 2.0 ** 7, 2.0 ** 8, 2.0 ** 9, 2.0 ** 10, 2.0 ** 11, 2.0 ** 12, 2.0 ** 13, 2.0 ** 14, 2.0 ** 15, 2.0 ** 16, 2.0 ** 17, 2.0 ** 18, 2.0 ** 19, 2.0 ** 20]
filepath = "data/hypothesis_2"

hypothesis = "HNSW quality remains steady while LSH quality increases with growing spread."

description = """
This test generates two clusters in the upper-right quadrant of the coordinate system.
The center points of the clusters are chosen to be $-spread / 2.0$ and $spread / 2.0$ respectively.
Both clusters allow for points within -0.2 and 0.2 range across all axes.
"""

np.random.seed(42)

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(spreads)
file.attrs['hypothesis'] = hypothesis
file.attrs['description'] = description

for i in range(len(spreads)):
    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    first_center = -(spreads[i] / 2.0)
    second_center = (spreads[i] / 2.0)
    first_cluster = np.random.uniform(first_center - 0.2, first_center + 0.2, (int(n_sites / 2), n_dims)) - 0.7
    second_cluster = np.random.uniform(second_center - 0.2, second_center + 0.2, (int(n_sites / 2), n_dims)) + 0.3
    sites = first_cluster + second_cluster
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')
    query = np.random.uniform(first_center, second_center, (1, n_dims))
    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims)
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')
    time_start = time.perf_counter()
    distance, solution = index.search(query, k)
    time_end = time.perf_counter()
    print(f'\tComputing solution: {time_end - time_start:.3f} seconds')

    instance = file.create_dataset('instance_' + str(i), data=sites)
    instance.attrs['n_sites'] = n_sites
    instance.attrs['n_dims'] = n_dims
    file.create_dataset('query_' + str(i), data=query)
    file.create_dataset('solution_' + str(i), data=solution)
    file.create_dataset('distance_' + str(i), data=distance)
