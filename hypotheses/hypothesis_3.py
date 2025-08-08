#!/usr/bin/env python3
import time
import numpy as np
import h5py
import faiss

# Given a list that contains the ids of sites, invert populates a new list where the value at the id is the rank of the site.
# If, for example, the site with id 5 was the nearest neighbor, then the resulting list would have value 1 at position 5.
# This way, computing the ranks of candidate solutions is possible in constant time, at the cost of some memory.
def invert(l):
    new_l = [0 for i in range(len(l))]

    for index, value in enumerate(l):
        new_l[value] = index + 1

    return new_l

# Hypothesis 3: HNSW remains precise on a uniform grid, whereas LSH degenerates due to cosine similarity collisions.
filepath = "data/hypothesis_3.h5"

hypothesis = "HNSW remains precise on a uniform grid, whereas LSH degenerates due to cosine similarity collisions."

description = """
This test generates a two-dimensional lattice with fixed extents.
To this extent, all sites have the form $(i, j)$ with $i, j \in \mathbb{N}$ and $i, j \leq \\text{extent}$.
"""

np.random.seed(42)

# Define all relevant data
extents = [(2 ** i) for i in range(4, 10)]
n_sites = [(extents[i] ** 2) for i in range(len(extents))]
n_dims = [2 for i in range(len(extents))]
n_planes = [(2 * dim) for dim in n_dims]
k = [5 for i in range(len(extents))]
sample_size = 20
site_generator = lambda i: [(np.float64(x),np.float64(y)) for x in range(extents[i]) for y in range(extents[i])]
query_generator = lambda i: np.random.uniform(0.0, extents[i] - 1.0, (sample_size, n_dims[i]))

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
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

    k_nearest = list(map(lambda x: x[:k[i]], solution))
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

file.attrs['var_name'] = "Extent"
file.attrs['var_values'] = extents
