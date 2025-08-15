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

GIST_DIM = 960
GIST_N = 1_000_000
GIST_QN = 1_000

sites = np.fromfile("/home/lazylambda/Downloads/gist/gist_base.fvecs", dtype=np.float32)
if sites.size == 0:
    raise Exception("Failed to load sites")

# Have an additional dimension in loading, as it contains the dimension information in the .fvecs file.
sites = sites.reshape(-1, GIST_DIM + 1)
# This trims of the dimension information
sites = sites[:, 1:]

# Do the same thing to extract the query vectors.
queries = np.fromfile("/home/lazylambda/Downloads/gist/gist_query.fvecs", dtype=np.float32)
if queries.size == 0:
    raise Exception("Failed to load query data")

queries = queries.reshape(-1, GIST_DIM + 1)
queries = queries[:, 1:]

sample_size = GIST_QN
n_sites = [GIST_N]
n_dims = [GIST_DIM]
n_planes = [GIST_DIM * 2]
k = [1000]
filepath = "data/gist1m.h5"
site_generator = lambda i: sites
query_generator = lambda i: queries[:sample_size, :]

hypothesis = ""

description = """"""

file = h5py.File(filepath, 'w')
file.attrs['k'] = k
file.attrs['n_dims'] = n_dims
file.attrs['n_sites'] = n_sites
file.attrs['n_planes'] = n_planes
file.attrs['n_instances'] = len(n_sites)
file.attrs['description'] = description
file.attrs['hypothesis'] = hypothesis
file.attrs['sample_size'] = sample_size
file.create_dataset('instance', data=sites)

index = faiss.IndexFlatL2(n_dims[0])
index.add(sites)

for chunk in range(sample_size // 20):
    print(f'Starting chunk {chunk + 1} of {sample_size // 20}')
    chunk_queries = queries[chunk * 20:(chunk + 1)*20]

    time_start = time.perf_counter()

    distance, solution = index.search(chunk_queries, n_sites[0])

    time_end = time.perf_counter()
    print(f'\tComputing solution: {time_end - time_start:.3f} seconds')

    k_nearest = list(map(lambda x: x[:k[0]], solution))
    ranks = solution
    print(ranks)
    print(len(ranks))

    ranks = list(map(invert, ranks))

    site_to_distance = []
    for sample_idx in range(20):
        print(f'Site-to-distance {sample_idx}')
        temp = [0.0 for i in range(n_sites[0])]
        for j in range(len(temp)):
            site_idx = solution[sample_idx][j]
            temp[site_idx] = distance[sample_idx][j]
        site_to_distance.append(temp)

    file.create_dataset('queries_' + str(chunk), data=chunk_queries)
    file.create_dataset('solution_' + str(chunk), data=k_nearest)
    file.create_dataset('ranks_' + str(chunk), data = ranks)
    file.create_dataset('distance_' + str(chunk), data=site_to_distance)
file.attrs['var_name'] = "Something"
file.attrs['var_values'] = k
