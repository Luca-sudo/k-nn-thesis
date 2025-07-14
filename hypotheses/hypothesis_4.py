#!/usr/bin/env python3

# Hypothesis 4: The observed loss of quality in hypothesis 3 can /not/ be counteracted by increasing the number of separating hyperplanes.


import time
import numpy as np
import h5py
import faiss

n_dims = 2
k = 5
extent = 100
n_planes = [i * n_dims for i in range(20)]
sample_size = 20
filepath = "data/hypothesis_4"

hypothesis = "The observed loss of quality in hypothesis 3 can /not/ be counteracted by increasing the number of separating hyperplanes."

description = """
Consider a uniform grid, akin to hypothesis 3, but this time with a fixed extent. Successive instances increase the number of separating hyperplanes.
"""

np.random.seed(42)

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(n_planes)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis
file.attrs['sample_size'] = sample_size
file.attrs['var_name'] = "Nr. of Hyperplanes"
file.attrs['var_values'] = n_planes

for i in range(len(n_planes)):

    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    sites = [(x,y) for x in range(extent) for y in range(extent)]
    sites = np.array(sites, dtype=np.float32)
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')

    queries = np.random.uniform(0, extent, (sample_size, n_dims))
    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims)
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    distance, solution = index.search(queries, extent ** 2)
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
    instance.attrs['n_sites'] = extent ** 2
    instance.attrs['n_dims'] = n_dims
    instance.attrs['n_planes'] = n_planes[i]
    file.create_dataset('queries_' + str(i), data=queries)
    file.create_dataset('solution_' + str(i), data=k_nearest)
    file.create_dataset('ranks_' + str(i), data = ranks)
    file.create_dataset('distance_' + str(i), data=k_nearest_distances)
