#!/usr/bin/env python3

# Hypothesis 6: Quality of LSH queries degenerates as density of sites increases.


import itertools
import time
import numpy as np
import h5py
import faiss

n_sites = [100, 1000, 5000, 10000, 50000, 100000, 500000, 100000]
n_dims = [100 for i in range(len(n_sites))]
k = 5
filepath = "data/hypothesis_6"

hypothesis = "Quality of LSH queries degenerates as density of sites increases."

description = """
We generate a uniformly-distributed set of sites inside of the unit hypercube centered around the origin. Successive instances increase the number of sites linearly.
"""

np.random.seed(42)

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(n_dims)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis

for i in range(len(n_dims)):

    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    sites = np.random.uniform(-0.5, 0.5, (n_sites[i], n_dims[i]))
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')

    query = np.random.uniform(-0.5, 0.5, (1, n_dims[i]))
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
    instance.attrs['n_planes'] = n_dims[i] * 2
    file.create_dataset('query_' + str(i), data=query)
    file.create_dataset('solution_' + str(i), data=solution)
    file.create_dataset('distance_' + str(i), data=distance)
