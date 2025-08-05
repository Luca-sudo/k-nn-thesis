#!/usr/bin/env python3

# Hypothesis 2: HNSW quality remains steady while LSH quality improves with increased spread (due to higher cosine similarity).

import time
import numpy as np
import h5py
import faiss

filepath = "data/hypothesis_2.h5"

hypothesis = "HNSW quality remains steady while LSH quality increases with growing spread."

description = """
This test generates two clusters in the upper-right quadrant of the coordinate system.
The center points of the clusters are chosen to be $-spread / 2.0$ and $spread / 2.0$ respectively.
Both clusters allow for points within -0.2 and 0.2 range across all axes.
"""

np.random.seed(42)

n_sites = 100000
n_dims = 100
k = 5
sample_size = 20

# This includes spreads up until (and including) $2^{20}$.
spreads = [2.0 ** i for i in range(21)]

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(spreads)
file.attrs['hypothesis'] = hypothesis
file.attrs['description'] = description
file.attrs['sample_size'] = sample_size
file.attrs['var_name'] = 'Spread'
file.attrs['var_values'] = spreads

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

    queries = np.random.uniform(first_center, second_center, (sample_size, n_dims))

    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims)
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')
    time_start = time.perf_counter()
    distance, solution = index.search(queries, n_sites)
    time_end = time.perf_counter()
    print(f'\tComputing solution: {time_end - time_start:.3f} seconds')

    k_nearest = list(map(lambda x: x[:k], solution))
    ranks = solution
    k_nearest_distances = list(map(lambda x: x[:k], distance))

    def invert(l):
        new_l = [0 for i in range(len(l))]

        for index, value in enumerate(l):
            new_l[value] = index + 1

        return new_l

    ranks = list(map(invert, ranks))

    instance = file.create_dataset('instance_' + str(i), data=sites)
    instance.attrs['n_sites'] = n_sites
    instance.attrs['n_dims'] = n_dims
    instance.attrs['n_planes'] = n_dims * 2
    file.create_dataset('queries_' + str(i), data=queries)
    file.create_dataset('solution_' + str(i), data=k_nearest)
    file.create_dataset('ranks_' + str(i), data = ranks)
    file.create_dataset('distance_' + str(i), data=k_nearest_distances)
