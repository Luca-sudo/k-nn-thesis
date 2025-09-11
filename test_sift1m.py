#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import time
import numpy as np
import h5py
import faiss
from functools import reduce

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to compare!")

filepath = sys.argv[1]

CHUNK_SIZE = 20
file = h5py.File(filepath, 'r')
n_instances = file.attrs['n_instances'].astype(int)
k = [i for i in range(10, 1010, 10)]
n_sites = file.attrs['n_sites'].astype(int).tolist()
description = file.attrs['description']
hypothesis = file.attrs['hypothesis']
sample_size = file.attrs['sample_size']
n_dims = file.attrs['n_dims'].astype(int).tolist()
n_planes = file.attrs['n_planes'].astype(int).tolist()


print(f'Hypothesis: {hypothesis}')
print(f'Description: {description}')

timings = []
qualities = []
ranks = []
recalls = []


print(f'Extracting instance.')
time_start = time.perf_counter()
sites = file['instance'][()]
time_end = time.perf_counter()
dt = round(time_end - time_start, 7)
print(f'\tExtracting data: \t{dt:.3f} seconds')

time_start = time.perf_counter()
hnsw_index = faiss.IndexHNSWFlat(n_dims[0], 32)
hnsw_index.hnsw.efConstruction = 40
hnsw_index.add(sites)
time_end = time.perf_counter()
dt = round(time_end - time_start, 7)
print(f'\tCreating HNSW Index: \t{time_end - time_start:.3f} seconds')
timings.append({
    'instance' : 0,
    'algo' : 'hnsw',
    'event' : 'creation',
    'dt' : dt
})

time_start = time.perf_counter()
lsh_index = faiss.IndexLSH(n_dims[0], n_planes[0])
lsh_index.add(sites)
time_end = time.perf_counter()
dt = round(time_end - time_start, 7)
print(f'\tCreating LSH Index: \t{time_end - time_start:.3f} seconds')
timings.append({
    'instance' : 0,
    'algo' : 'lsh',
    'event' : 'creation',
    'dt' : dt
})

for i in range(n_instances):

    site_to_distance = file['distance_' + str(i)][()]
    site_to_rank = file['ranks_' + str(i)][()]
    queries = file['queries_' + str(i)][()]
    solution = file['solution_' + str(i)][()]

    for k_i in k:

        time_start = time.perf_counter()
        hnsw_index.hnsw.efSearch = 32
        _, hnsw_solutions = hnsw_index.search(queries, k_i)

        time_end = time.perf_counter()
        dt = round(time_end - time_start, 7)
        print(f'\tQuerying HNSW Index: \t{time_end - time_start:.3f} seconds')
        timings.append({
            'instance' : 0,
            'algo' : 'hnsw',
            'event' : 'query',
            'dt' : dt / CHUNK_SIZE
        })


        time_start = time.perf_counter()
        _, lsh_solutions = lsh_index.search(queries, k_i)
        time_end = time.perf_counter()
        dt = round(time_end - time_start, 7)
        print(f'\tQuerying LSH Index: \t{time_end - time_start:.3f} seconds')
        timings.append({
            'instance' : 0,
            'algo' : 'lsh',
            'event' : 'query',
            'dt' : dt / CHUNK_SIZE
        })

        solution_vectors = np.array(list(map(lambda x : sites[x], lsh_solutions)))
        lsh_distance = []
        for sample_idx in range(CHUNK_SIZE):
            sample = solution_vectors[sample_idx, :, :]
            distances = sorted(np.sum((sample - queries[sample_idx]) ** 2, axis=1))
            lsh_distance.append(distances)

        lsh_quality = []
        hnsw_quality = []
        for sample_idx in range(CHUNK_SIZE):
            lsh_quality.append(site_to_distance[sample_idx][solution[sample_idx][:k_i]] / sorted(site_to_distance[sample_idx][lsh_solutions[sample_idx][:k_i]]))
            hnsw_quality.append(site_to_distance[sample_idx][solution[sample_idx][:k_i]] / sorted(site_to_distance[sample_idx][hnsw_solutions[sample_idx][:k_i]]))

        print(f'hnsw_quality {hnsw_quality}')
        print(f'lsh_quality {lsh_quality}')

        var_name = file.attrs['var_name']
        var_value = file.attrs['var_values'][i]

        for sample in lsh_quality:
            for q in sample:
                    qualities.append({
                    var_name : var_value,
                    'algo' : 'lsh',
                    'Quality' : q
                    })

        for sample in hnsw_quality:
            for q in sample:
                    qualities.append({
                    var_name : var_value,
                    'algo' : 'hnsw',
                    'Quality' : q
                    })

        for sample_idx, neighbors in enumerate(lsh_solutions):
            for r in site_to_rank[sample_idx][neighbors]:
                    ranks.append({
                            var_name : var_value,
                            'algo' : 'lsh',
                            'Rank' : r
                    })

        for sample_idx, neighbors in enumerate(hnsw_solutions):
            for r in site_to_rank[sample_idx][neighbors]:
                    ranks.append({
                            var_name : var_value,
                            'algo' : 'hnsw',
                            'Rank' : r
                    })

        for sample_idx, neighbors in enumerate(lsh_solutions):
            neighbor_count = 0
            for r in site_to_rank[sample_idx][neighbors]:
                if r <= k_i:
                    neighbor_count += 1
            recalls.append({
                    var_name : var_value,
                    'algo' : 'lsh',
                    'Recall' : neighbor_count / k_i,
                    'k' : k_i
            })

        for sample_idx, neighbors in enumerate(hnsw_solutions):
            neighbor_count = 0
            for r in site_to_rank[sample_idx][neighbors]:
                if r <= k_i:
                    neighbor_count += 1
            recalls.append({
                    var_name : var_value,
                    'algo' : 'hnsw',
                    'Recall' : neighbor_count / k_i,
                    'k' : k_i
            })

timings = pd.DataFrame(timings)
qualities = pd.DataFrame(qualities)
ranks = pd.DataFrame(ranks)
recalls = pd.DataFrame(recalls)

metadata = {
    'var_name' : file.attrs['var_name']
}

target_path = filepath[:-3]

timings.to_hdf(target_path + "_results.h5", key="timings", format="table")
qualities.to_hdf(target_path + "_results.h5", key="qualities", format="table")
ranks.to_hdf(target_path + "_results.h5", key="ranks", format="table")
recalls.to_hdf(target_path + "_results.h5", key="recalls", format="table")

results = h5py.File(target_path + "_results.h5", 'r+')

results.attrs['var_name'] = file.attrs['var_name']
results.attrs['var_values'] = file.attrs['var_values']
