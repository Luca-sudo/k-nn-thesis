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

# Hypothesis 2: HNSW quality remains steady while LSH quality improves with increased spread (due to higher cosine similarity).

filepath = "data/hypothesis_2.h5"

hypothesis = "HNSW quality remains steady while LSH quality increases with growing spread."

description = """
This test generates two clusters in the upper-right quadrant of the coordinate system.
The center points of the clusters are chosen to be $1.0$ and $1.0 + spread$ respectively.
Both clusters allow for points within -0.2 and 0.2 range across all axes.
"""

np.random.seed(42)

spreads = [2.0 ** i for i in range(1, 15)]
n_sites = [10000 for i in range(len(spreads))]
n_dims = [100 for i in range(len(spreads))]
n_planes = [2 * dim for dim in n_dims]
max_k = 100
sample_size = 22
first_center = [3.0 for i in spreads]
second_center = [first_center[i] + spreads[i] for i in range(len(spreads))]
print(f'first_center: {first_center}')
print(f'second_center: {second_center}')
site_generator = lambda iteration: np.concatenate([np.random.uniform(first_center[iteration] - 2.0, first_center[iteration] + 2.0, (int(n_sites[iteration] / 2), n_dims[iteration])), np.random.uniform(second_center[iteration] - 2.0, second_center[iteration] + 2.0, (int(n_sites[iteration] / 2), n_dims[iteration]))])
query_generator = lambda iteration: np.concatenate([np.random.uniform(first_center[iteration] - 2.0, first_center[iteration] + 2.0, (int(sample_size / 2), n_dims[iteration])), np.random.uniform(second_center[iteration] - 2.0, second_center[iteration] + 2.0, (int(sample_size / 2), n_dims[iteration]))])


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
    queries = np.array(queries, dtype=np.float32)
    print(f'Iteration {i}; queries {queries}')
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

file.attrs['var_name'] = "Spread"
file.attrs['var_values'] = spreads
