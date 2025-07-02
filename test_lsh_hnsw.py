#!/usr/bin/env python3
import matplotlib.pyplot as plt
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
    n_planes = dataset.attrs['n_planes'].item()
    print('n_planes: ', n_planes)
    sites = dataset[()]
    optimal_distances = file['distance_' + str(i)][()]
    ordered_sites = file['ordered_sites_' + str(i)][()]
    query = file['query_' + str(i)][()]
    solution = file['solution_' + str(i)][()]
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    data_extraction.append(dt)
    print(f'\tExtracting data: \t{dt:.3f} seconds')

    time_start = time.perf_counter()
    hnsw_index = faiss.IndexHNSWFlat(n_dim, 4)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.add(sites)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    hnsw_creation.append(dt)
    print(f'\tCreating HNSW Index: \t{time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    hnsw_distance, hnsw_solutions = hnsw_index.search(query, k)
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    hnsw_creation.append(dt)
    print(f'\tQuerying HNSW Index: \t{time_end - time_start:.3f} seconds')

    time_start = time.perf_counter()
    lsh_index = faiss.IndexLSH(n_dim, n_planes)
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
    lsh_distance = np.sum((solution_vectors[0] - query) ** 2, axis=1)

    sorted_lsh_distances = sorted(lsh_distance)
    sorted_hnsw_distances = sorted(hnsw_distance[0])

    lsh_quality = optimal_distances / sorted_lsh_distances
    hnsw_quality = optimal_distances / sorted_hnsw_distances

    print('\tLSH Min, Median, Max:\t\t', min(lsh_quality), lsh_quality[1], max(lsh_quality))
    print('\tHNSW Min, Median, Max:\t\t', min(hnsw_quality), hnsw_quality[1], max(hnsw_quality))

    hnsw_min.append(min(hnsw_quality))
    lsh_min.append(min(lsh_quality))
    hnsw_median.append(hnsw_quality[1])
    lsh_median.append(lsh_quality[1])
    hnsw_max.append(max(hnsw_quality))
    lsh_max.append(max(lsh_quality))

    lsh_neighbor_order = ordered_sites[lsh_solutions]
    hnsw_neighbor_order = ordered_sites[hnsw_solutions]

    print('\tLSH Neighbor Order: ', lsh_neighbor_order)
    print('\tHNSW Neighbor Order: ', hnsw_neighbor_order)

rows = zip(data_extraction, hnsw_creation, hnsw_querying, lsh_creation, lsh_querying, hnsw_min, hnsw_median, hnsw_max, lsh_min, lsh_median, lsh_max)

csv_path = filepath + ".csv"

with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

helper = [2**i for i in range(n_instances)]

fig, (p1, p2, p3) = plt.subplots(1, 3)

p1.set_title('Minimum Quality')
p1.plot(helper, lsh_min[1:], 'b', helper, hnsw_min[1:], 'g')
p1.set_xscale('log', base=2)

p2.set_title('Median Quality')
p2.plot(helper, lsh_median[1:], 'b', helper, hnsw_median[1:], 'g')
p2.set_xscale('log', base=2)

p3.set_title('Maximum Quality')
p3.plot(helper, lsh_max[1:], 'b', helper, hnsw_max[1:], 'g')
p3.set_xscale('log', base=2)


plt.show()
