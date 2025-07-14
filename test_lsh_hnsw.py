#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import time
import numpy as np
import h5py
import faiss

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to compare!")

filepath = sys.argv[1]

file = h5py.File(filepath, 'r')
k = file.attrs['k'].item()
n_instances = file.attrs['n_instances']
description = file.attrs['description']
hypothesis = file.attrs['hypothesis']
sample_size = file.attrs['sample_size']


print(f'Hypothesis: {hypothesis}')
print(f'Description: {description}')

timings = []
qualities = []
ranks = []

for i in range(n_instances):
    print(f'Extracting instance {i}.')
    time_start = time.perf_counter()
    dataset = file['instance_' + str(i)]
    n_dim = dataset.attrs['n_dims'].item()
    n_sites = dataset.attrs['n_sites']
    n_planes = dataset.attrs['n_planes'].item()
    print('n_planes: ', n_planes)
    sites = dataset[()]
    optimal_distances = file['distance_' + str(i)][()]
    site_to_rank = file['ranks_' + str(i)][()]
    queries = file['queries_' + str(i)][()]
    solution = file['solution_' + str(i)][()]
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tExtracting data: \t{dt:.3f} seconds')

    time_start = time.perf_counter()
    hnsw_index = faiss.IndexHNSWFlat(n_dim, 4)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.add(sites)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tCreating HNSW Index: \t{time_end - time_start:.3f} seconds')
    timings.append({
        'instance' : i,
        'algo' : 'hnsw',
        'event' : 'creation',
        'dt' : dt
    })

    time_start = time.perf_counter()
    hnsw_distance, hnsw_solutions = hnsw_index.search(queries, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tQuerying HNSW Index: \t{time_end - time_start:.3f} seconds')
    timings.append({
        'instance' : i,
        'algo' : 'hnsw',
        'event' : 'query',
        'dt' : dt / sample_size
    })

    time_start = time.perf_counter()
    lsh_index = faiss.IndexLSH(n_dim, n_planes)
    lsh_index.add(sites)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tCreating LSH Index: \t{time_end - time_start:.3f} seconds')
    timings.append({
        'instance' : i,
        'algo' : 'lsh',
        'event' : 'creation',
        'dt' : dt
    })

    time_start = time.perf_counter()
    _, lsh_solutions = lsh_index.search(queries, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tQuerying LSH Index: \t{time_end - time_start:.3f} seconds')
    timings.append({
        'instance' : i,
        'algo' : 'lsh',
        'event' : 'query',
        'dt' : dt / sample_size
    })

    print(f'\tLSH Solutions: {lsh_solutions}')
    print(f'\tHNSW Solutions: {hnsw_solutions}')

    solution_vectors = np.array(list(map(lambda x : sites[x], lsh_solutions)))
    lsh_distance = []
    for sample_idx in range(sample_size):
        print(sample_idx)
        sample = solution_vectors[sample_idx, :, :]
        distances = sorted(np.sum((sample - queries[sample_idx]) ** 2, axis=1))
        lsh_distance.append(distances)

    sorted_lsh_distances = lsh_distance
    sorted_hnsw_distances = list(map(lambda x : sorted(x), hnsw_distance))
    print(f'hnsw_distance: {hnsw_distance}')
    print(f'hnsw_sorted_distance: {sorted_hnsw_distances}')

    lsh_quality = optimal_distances / sorted_lsh_distances
    print(f'lsh_quality {lsh_quality}')
    hnsw_quality = optimal_distances / sorted_hnsw_distances
    print(f'hnsw_quality {hnsw_quality}')

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

    current_lsh_ranks = []
    current_hnsw_ranks = []

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

timings = pd.DataFrame(timings)
qualities = pd.DataFrame(qualities)
ranks = pd.DataFrame(ranks)

metadata = {
    'var_name' : file.attrs['var_name']
}

timings.to_hdf(filepath + "_results.h5", key="timings", format="table")
qualities.to_hdf(filepath + "_results.h5", key="qualities", format="table")
ranks.to_hdf(filepath + "_results.h5", key="ranks", format="table")

results = h5py.File(filepath + "_results.h5", 'r+')

results.attrs['var_name'] = file.attrs['var_name']
