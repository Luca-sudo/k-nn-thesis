#!/usr/bin/env python3

# Hypothesis 1: HNSW remains more precise than LSH on clustered data as the number of sites grows.

import time
import numpy as np
import h5py
import faiss

filepath = "data/hypothesis_1.h5"

np.random.seed(42)

hypothesis = "HNSW remains more precise than LSH on clustered data as the number of sites grows."

description = '''
We generate two clusters, centered at $(-0.5, \dots, -0.5)$ and $(0.5, \dots, 0.5)$ respectively. These are hypercubes with a diameter of $0.4$. Put differently, for all sites $s = (s_1, \dots, s_d)$ in the first cluster, it holds that $-0.7 \leq s_i \leq -0.3$. For the second cluster this corresponds to $0.3 \leq s_i \leq 0.7$. Note that sites within each cluster are sampled uniformly.
'''

n_sites = [2**i for i in range(11, 23)]
n_dims = [100 for i in range(11, 23)]
k = 5
sample_size = 20

assert(len(n_sites) == len(n_dims))

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(n_sites)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis
file.attrs['sample_size'] = sample_size
file.attrs['var_name'] = 'Number of Sites'
file.attrs['var_values'] = n_sites

for i in range(len(n_sites)):

    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    first_cluster = np.random.uniform(-0.7, -0.3, (int(n_sites[i] / 2), n_dims[i]))
    second_cluster = np.random.uniform(0.3, 0.7, (int(n_sites[i] / 2), n_dims[i]))
    sites = first_cluster + second_cluster
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')

    queries = np.random.uniform(-1.0, 1.0, (sample_size, n_dims[i]))

    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims[i])
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')
    time_start = time.perf_counter()
    distance, solution = index.search(queries, int(n_sites[i]))
    time_end = time.perf_counter()
    print(f'\tComputing solution: {time_end - time_start:.3f} seconds')
    print(f'\tDistances: {distance}')
    print(f'\tSolutions: {solution}')

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
    instance.attrs['n_sites'] = n_sites[i]
    instance.attrs['n_dims'] = n_dims[i]
    instance.attrs['n_planes'] = n_dims[i] * 2
    file.create_dataset('queries_' + str(i), data=queries)
    file.create_dataset('solution_' + str(i), data=k_nearest)
    file.create_dataset('ranks_' + str(i), data = ranks)
    file.create_dataset('distance_' + str(i), data=k_nearest_distances)
