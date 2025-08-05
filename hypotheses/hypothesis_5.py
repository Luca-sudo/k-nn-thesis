#!/usr/bin/env python3

# Hypothesis 5: Over a uniform-grid similar to hypotheses 3 & 4, the quality of LSH diminishes as the dimensionality of the feature space increases over a bounded region.

import itertools
import time
import numpy as np
import h5py
import faiss

n_dims = [i for i in range(2, 11)]
k = 5
extent = 5
sample_size = 20
filepath = "data/hypothesis_5.h5"

hypothesis = "Over a uniform-grid similar to hypotheses 3 & 4, the quality of LSH diminishes as the dimensionality of the feature space increases."

description = """
Consider a uniform grid, akin to hypothesis 3 & 4, but this time the extent is fixed and the dimensionality increase for successive instances.
"""

np.random.seed(42)

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_instances'] = len(n_dims)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis
file.attrs['sample_size'] = sample_size
file.attrs['var_name'] = "Dimension"
file.attrs['var_values'] = n_dims

for i in range(len(n_dims)):

    print(f'Generating instance {i}:')
    time_start = time.perf_counter()
    sites = [s for s in itertools.product(range(extent), repeat=n_dims[i])]
    sites = np.array(sites, dtype=np.float32)
    time_end = time.perf_counter()
    print(f'\tGenerating sites: {time_end - time_start:.3f} seconds')

    queries = np.random.uniform(0, extent, (sample_size, n_dims[i]))
    time_start = time.perf_counter()
    index = faiss.IndexFlatL2(n_dims[i])
    index.add(sites)
    time_end = time.perf_counter()
    print(f'\tGenerating flat index: {time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    distance, solution = index.search(queries, extent ** n_dims[i])
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
    instance.attrs['n_sites'] = extent ** n_dims[i]
    instance.attrs['n_dims'] = n_dims[i]
    instance.attrs['n_planes'] = n_dims[i] * 2
    file.create_dataset('queries_' + str(i), data=queries)
    file.create_dataset('solution_' + str(i), data=k_nearest)
    file.create_dataset('ranks_' + str(i), data = ranks)
    file.create_dataset('distance_' + str(i), data=k_nearest_distances)
