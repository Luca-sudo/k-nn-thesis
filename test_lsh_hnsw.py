#!/usr/bin/env python3

import csv
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

data_extraction = ["Data extraction in seconds"]
hnsw_creation = ["Creation of HNSW index in seconds"]
hnsw_querying = ["Querying of HNSW index in seconds"]
lsh_creation = ["Creation of LSH index in seconds"]
lsh_querying = ["Querying of LSH index in seconds"]
hnsw_min = ["HNSW Min Quality"]
hnsw_median = ["HNSW Median Quality"]
hnsw_max = ["HNSW Max Quality"]
lsh_min = ["LSH Min Quality"]
lsh_median = ["LSH Median Quality"]
lsh_max = ["LSH Max Quality"]

for i in range(n_instances):
    print(f'Extracting instance {i}.')
    time_start = time.perf_counter()
    dataset = file['instance_' + str(i)]
    n_dim = dataset.attrs['n_dims'].item()
    n_sites = dataset.attrs['n_sites']
    sites = dataset[()]
    optimal_distances = file['distance_' + str(i)][()]
    query = file['query_' + str(i)][()]
    solution = file['solution_' + str(i)][()]
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    data_extraction.append(dt)
    print(f'\tExtracting data: \t{dt:.3f} seconds')

    time_start = time.perf_counter()
    hnsw_index = faiss.IndexHNSWFlat(n_dim, 16)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.add(sites)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    hnsw_creation.append(dt)
    print(f'\tCreating HNSW Index: \t{time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    hnsw_distance, queried_solution = hnsw_index.search(query, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    hnsw_creation.append(dt)
    print(f'\tQuerying HNSW Index: \t{time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    lsh_index = faiss.IndexLSH(n_dim, 2 * n_dim)
    lsh_index.add(sites)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    hnsw_creation.append(dt)
    print(f'\tCreating LSH Index: \t{time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    _, lsh_solutions = lsh_index.search(query, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    hnsw_creation.append(dt)
    print(f'\tQuerying LSH Index: \t{time_end - time_start:.3f} seconds')

    solution_vectors = sites[lsh_solutions]
    lsh_distance = np.round(np.sum((solution_vectors[0] - query) ** 2, axis=1), decimals=7)

    lsh_quality = np.round(optimal_distances / lsh_distance, decimals=7)
    hnsw_quality = optimal_distances / hnsw_distance

    lsh_quality = sorted(lsh_quality[0], reverse=True)
    hnsw_quality = sorted(hnsw_quality[0], reverse=True)

    print('\tLSH Min, Median, Max:\t\t', min(lsh_quality), round(sum(lsh_quality)/k, 7), max(lsh_quality))
    print('\tHNSW Min, Median, Max:\t\t', min(hnsw_quality), round(sum(hnsw_quality)/k, 7), max(hnsw_quality))

    hnsw_min.append(min(hnsw_quality))
    lsh_min.append(min(lsh_quality))
    hnsw_median.append(hnsw_quality[1])
    lsh_median.append(lsh_quality[1])
    hnsw_max.append(max(hnsw_quality))
    lsh_max.append(max(lsh_quality))


rows = zip(data_extraction, hnsw_creation, hnsw_querying, lsh_creation, lsh_querying, hnsw_min, hnsw_median, hnsw_max, lsh_min, lsh_median, lsh_max)

csv_path = filepath + ".csv"

with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
