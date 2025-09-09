#!/usr/bin/env python3
import time
import numpy as np
import h5py
import faiss
import sys

# Given a list that contains the ids of sites, invert populates a new list where the value at the id is the rank of the site.
# If, for example, the site with id 5 was the nearest neighbor, then the resulting list would have value 1 at position 5.
# This way, computing the ranks of candidate solutions is possible in constant time, at the cost of some memory.
def invert(l):
    new_l = [0 for i in range(len(l))]

    for index, value in enumerate(l):
        new_l[value] = index + 1

    return new_l

# Hypothesis 1: HNSW remains more precise than LSH on clustered data as the number of sites grows.
filepath = "data/hypothesis_1.h5"

np.random.seed(42)

hypothesis = "HNSW remains more precise than LSH on clustered data as the number of sites grows."

description = '''
We generate two clusters, centered at $(-0.5, \dots, -0.5)$ and $(0.5, \dots, 0.5)$ respectively. These are hypercubes with a diameter of $0.4$. Put differently, for all sites $s = (s_1, \dots, s_d)$ in the first cluster, it holds that $-0.7 \leq s_i \leq -0.3$. For the second cluster this corresponds to $0.3 \leq s_i \leq 0.7$. Note that sites within each cluster are sampled uniformly.
'''

n_sites = [2**i for i in range(11, 23)]
n_dims = [100 for i in n_sites]
n_planes = [dim * 2 for dim in n_dims]
max_k = 100
sample_size = 30
site_generator = lambda i: np.concat([np.random.uniform(-0.7, -0.3, (int(n_sites[i] / 2), n_dims[i])), np.random.uniform(0.3, 0.7, (int(n_sites[i] / 2), n_dims[i]))])
query_generator = lambda i: np.random.uniform(-1.0, 1.0, (sample_size, n_dims[i]))

file = h5py.File(filepath, 'w')
file.attrs['max_k'] = max_k
file.attrs['n_dims'] = n_dims
file.attrs['n_sites'] = n_sites
file.attrs['n_planes'] = n_planes
file.attrs['n_instances'] = len(n_sites)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis
file.attrs['sample_size'] = sample_size

for i in range(len(n_sites)):
    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    sites = site_generator(i)
    sites = np.array(sites, dtype=np.float32)
    print(f'sites: {sites}')
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')
    queries = query_generator(i)
    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims[i])
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')


    time_start = time.perf_counter()

    distance, solution = index.search(queries, n_sites[i])

    time_end = time.perf_counter()
    print(f'\tComputing solution: {time_end - time_start:.3f} seconds')

    k_nearest = list(map(lambda x: x[:max_k], solution))
    ranks = solution
    print(ranks[0])
    print(len(ranks[0]))

    ranks = list(map(invert, ranks))

    site_to_distance = []
    for sample_idx in range(sample_size):
        temp = [0.0 for i in range(n_sites[i])]
        for j in range(len(temp)):
            site_idx = solution[sample_idx][j]
            temp[site_idx] = distance[sample_idx][j]
        site_to_distance.append(temp)

    instance = file.create_dataset('instance_' + str(i), data=sites)
    file.create_dataset('queries_' + str(i), data=queries)
    file.create_dataset('solution_' + str(i), data=k_nearest)
    file.create_dataset('ranks_' + str(i), data = ranks)
    file.create_dataset('distance_' + str(i), data=site_to_distance)

file.attrs['var_name'] = "Cluster Size"
file.attrs['var_values'] = list(map(lambda x: x / 2, n_sites))
