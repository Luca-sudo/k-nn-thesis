#!/usr/bin/env python3

# Hypothesis 1: HNSW remains more precise than LSH on clustered data as the number of sites grows.

import time
import numpy as np
import h5py
import faiss

filepath = "data/hypothesis_1"

np.random.seed(42)

hypothesis = "HNSW remains more precise than LSH on clustered data as the number of sites grows."

description = '''
We generate two clusters, centered at $(-0.5, \dots, -0.5)$ and $(0.5, \dots, 0.5)$ respectively. These are hypercubes with a diameter of $0.4$. Put differently, for all sites $s = (s_1, \dots, s_d)$ in the first cluster, it holds that $-0.7 \leq s_i \leq -0.3$. For the second cluster this corresponds to $0.3 \leq s_i \leq 0.7$. Note that sites within each cluster are sampled uniformly.

'''

n_sites = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
n_dims = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
k = 5

assert(len(n_sites) == len(n_dims))

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(n_sites)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis

for i in range(len(n_sites)):

    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    first_cluster = np.random.uniform(-0.7, -0.3, (int(n_sites[i] / 2), n_dims[i]))
    second_cluster = np.random.uniform(0.3, 0.7, (int(n_sites[i] / 2), n_dims[i]))
    sites = first_cluster + second_cluster
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')

    query = np.random.uniform(-1.0, 1.0, (1, n_dims[i]))

    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims[i])
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')
    time_start = time.perf_counter()
    distance, solution = index.search(query, k)
    time_end = time.perf_counter()
    print(f'\tComputing solution: {time_end - time_start:.3f} seconds')

    instance = file.create_dataset('instance_' + str(i), data=sites)
    instance.attrs['n_sites'] = n_sites[i]
    instance.attrs['n_dims'] = n_dims[i]
    file.create_dataset('query_' + str(i), data=query)
    file.create_dataset('solution_' + str(i), data=solution)
    file.create_dataset('distance_' + str(i), data=distance)
