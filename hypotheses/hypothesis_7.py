#!/usr/bin/env python3

# Hypothesis 7: LSH-quality and recall are servicable on a grid translated by an irrational number, because co-linearity of sites is curbed.

import time
import numpy as np
import h5py
import faiss

n_dims = 2
k = 5
extents = [5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 500]
sample_size = 20
filepath = "data/hypothesis_7.h5"

hypothesis = "LSH-quality and recall are servicable on a grid translated by an irrational number, because co-linearity of sites is curbed."

description = """
This test generates a two-dimensional lattice with fixed extents that is translated by $\pi$.
To this extent, all sites have the form $(i + \pi, j + \pi)$ with $i, j \in \mathbb{N}$ and $i, j \leq \\text{extent}$.
"""

np.random.seed(42)

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(extents)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis
file.attrs['sample_size'] = sample_size
file.attrs['var_name'] = "Extent"
file.attrs['var_values'] = extents

for i in range(len(extents)):
    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    sites = [(x + np.pi ,y + np.pi) for x in range(extents[i]) for y in range(extents[i])]
    sites = np.array(sites, dtype=np.float32)
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')
    queries = np.random.uniform(0, extents[i], (sample_size, n_dims))
    print(f'queries: {queries}')
    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims)
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    distance, solution = index.search(queries, extents[i] ** 2)
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
    instance.attrs['n_sites'] = extents[i] ** 2
    instance.attrs['n_dims'] = n_dims
    instance.attrs['n_planes'] = n_dims * 2
    file.create_dataset('queries_' + str(i), data=queries)
    file.create_dataset('solution_' + str(i), data=k_nearest)
    file.create_dataset('ranks_' + str(i), data = ranks)
    file.create_dataset('distance_' + str(i), data=k_nearest_distances)
