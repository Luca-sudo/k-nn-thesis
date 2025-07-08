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
    ordered_sites = file['ordered_sites_' + str(i)][()]
    query = file['query_' + str(i)][()]
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
    hnsw_distance, hnsw_solutions = hnsw_index.search(query, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tQuerying HNSW Index: \t{time_end - time_start:.3f} seconds')
    timings.append({
        'instance' : i,
        'algo' : 'hnsw',
        'event' : 'query',
        'dt' : dt
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
    _, lsh_solutions = lsh_index.search(query, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tQuerying LSH Index: \t{time_end - time_start:.3f} seconds')
    timings.append({
        'instance' : i,
        'algo' : 'lsh',
        'event' : 'query',
        'dt' : dt
    })

    solution_vectors = sites[lsh_solutions]
    lsh_distance = np.sum((solution_vectors[0] - query) ** 2, axis=1)

    sorted_lsh_distances = sorted(lsh_distance)
    sorted_hnsw_distances = sorted(hnsw_distance[0])

    lsh_quality = optimal_distances / sorted_lsh_distances
    hnsw_quality = optimal_distances / sorted_hnsw_distances

    print('\tLSH Min, Median, Max:\t\t', min(lsh_quality), lsh_quality[1], max(lsh_quality))
    print('\tHNSW Min, Median, Max:\t\t', min(hnsw_quality), hnsw_quality[1], max(hnsw_quality))

    for q in lsh_quality:
        qualities.append({
            'instance' : i,
            'algo' : 'lsh',
            'value' : q
        })

    for q in hnsw_quality:
        qualities.append({
            'instance' : i,
            'algo' : 'hnsw',
            'value' : q
        })

    current_lsh_ranks = ordered_sites[lsh_solutions]
    current_hnsw_ranks = ordered_sites[hnsw_solutions]

    for r in current_lsh_ranks[0]:
        ranks.append({
            'instance' : i,
            'algo' : 'lsh',
            'value' : r
        })

    for r in current_hnsw_ranks[0]:
        ranks.append({
            'instance' : i,
            'algo' : 'hnsw',
            'value' : r
        })

    print('\tLSH Ranks: ', current_lsh_ranks)
    print('\tHNSW Ranks: ', current_hnsw_ranks)

timings = pd.DataFrame(timings)
qualities = pd.DataFrame(qualities)
ranks = pd.DataFrame(ranks)

timings.to_csv(filepath + "_timings.csv")
qualities.to_csv(filepath + "_qualities.csv")
ranks.to_csv(filepath + "_ranks.csv")
