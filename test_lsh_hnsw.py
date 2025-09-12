#!/usr/bin/env python3
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import sys
import time
import numpy as np
import h5py
import faiss
from annoy import AnnoyIndex
import hnswlib
from functools import reduce
from sklearn.neighbors import KDTree, BallTree
from enum import Enum

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to compare!")

filepath = sys.argv[1]

file = h5py.File(filepath, 'r')
max_k = 110
k = [i for i in range(10, max_k, 10)]
n_instances = file.attrs['n_instances'].astype(int).tolist()
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

ef_search_factors = {
    10 : 40,
    20 : 20,
    30 : 15,
    40 : 10,
    50 : 10,
    60 : 10,
    70 : 10,
    80 : 10,
    90 : 10,
}

k_search_factors = {
    10 : 40,
    20 : 20,
    30 : 15,
    40 : 10,
    50 : 10,
    60 : 10,
    70 : 10,
    80 : 10,
    90 : 10,
}

class DS(Enum):
    LSH = 1
    HNSW = 2
    KD = 3
    BT = 4
    RS = 5
    BRUTE = 6

def BT(sample_size):
    return (DS.BT, sample_size)

def KD(sample_size):
    return (DS.KD, sample_size)

def HNSW(sample_size):
    return (DS.HNSW, sample_size)

def LSH(sample_size):
    return (DS.LSH, sample_size)

# Sample size doesnt matter for random sampling.
def RS(sample_size):
    return (DS.RS, sample_size)

def BRUTE(sample_size):
    return (DS.BRUTE, sample_size)

def to_string(ds):
    match ds[0]:
        case DS.LSH:
            return 'lsh'
        case DS.HNSW:
            return 'hnsw'
        case DS.KD:
            return f'kd@{ds[1]}'
        case DS.BT:
            return f'bt@{ds[1]}'
        case DS.RS:
            return f'rand sampling'
        case DS.BRUTE:
            return 'bruteforce'

ds_to_test = [
    LSH(1.0),
    HNSW(1.0),
    KD(0.2),
    KD(0.4),
    KD(0.6),
    KD(0.8),
    KD(1.0),
    BT(0.2),
    BT(0.4),
    BT(0.6),
    BT(0.8),
    BT(1.0),
    RS(1.0),
    BRUTE(1.0)
]

def create_index(ds, sites, n_dims, n_planes):
    # Maximum number of samples is hardcoded. Relevant for DS.RS
    MAX_SAMPLES = 100
    index = 0
    rand_samples = 0
    match ds[0]:
        case DS.LSH:
            time_start = time.perf_counter()
            index = AnnoyIndex(n_dims, 'euclidean')
            for j in range(len(sites)):
                index.add_item(j, sites[j])
            index.build(50)
            time_end = time.perf_counter()
            dt = round(time_end - time_start, 7)
            print(f'\tCreating LSH Index: \t{time_end - time_start:.3f} seconds')
            timings.append({
                'instance' : i,
                'algo' : 'lsh',
                'event' : 'creation',
                'dt' : dt
            })
        case DS.HNSW:
            time_start = time.perf_counter()
            index = hnswlib.Index(space='l2', dim=n_dims)
            index.init_index(max_elements=len(sites), ef_construction=200, M=16)
            index.add_items(sites)
            time_end = time.perf_counter()
            dt = round(time_end - time_start, 7)
            print(f'\tCreating HNSW Index: \t{time_end - time_start:.3f} seconds')
            timings.append({
                'instance' : i,
                'algo' : 'hnsw',
                'event' : 'creation',
                'dt' : dt
            })
        case DS.KD:
            time_start = time.perf_counter()
            rand_samples = np.random.choice(len(sites), int(len(sites) * ds[1]), replace=False)
            index = KDTree(sites[rand_samples], leaf_size=30, metric='euclidean')
            time_end = time.perf_counter()
            dt = round(time_end - time_start, 7)
            print(f'\tCreating KD-Tree@{ds[1]} Index: \t{time_end - time_start:.3f} seconds')
            timings.append({
                'instance' : i,
                'algo' : to_string(ds),
                'event' : 'creation',
                'dt' : dt
            })
        case DS.BT:
            time_start = time.perf_counter()
            rand_samples = np.random.choice(len(sites), int(len(sites) * ds[1]), replace=False)
            index = BallTree(sites[rand_samples], leaf_size=30, metric='euclidean')
            time_end = time.perf_counter()
            dt = round(time_end - time_start, 7)
            print(f'\tCreating Ball-Tree@{ds[1]} Index: \t{time_end - time_start:.3f} seconds')
            timings.append({
                'instance' : i,
                'algo' : to_string(ds),
                'event' : 'creation',
                'dt' : dt
            })
        case DS.RS:
            time_start = time.perf_counter()
            index = 0
            random_samples = 0
            time_end = time.perf_counter()
            dt = round(time_end - time_start, 7)
            print(f'\tRandom Sampling: \t{time_end - time_start:.3f} seconds')
            timings.append({
                'instance' : i,
                'algo' : to_string(ds),
                'event' : 'creation',
                'dt' : dt
            })
        case DS.BRUTE:
            time_start = time.perf_counter()
            random_sapmles = 0
            index = hnswlib.BFIndex(space='l2', dim=n_dims)
            index.init_index(max_elements=len(sites))
            index.add_items(sites, range(len(sites)))
            time_end = time.perf_counter()
            dt = round(time_end - time_start, 7)
            print(f'\tBruteforce: \t{time_end - time_start:.3f} seconds')
            timings.append({
                'instance' : i,
                'algo' : to_string(ds),
                'event' : 'creation',
                'dt' : dt
            })

    return (index, rand_samples)

def search_index(index, ds, rand_samples, queries, k_i, instance_num):
    index_solutions = 0
    match ds[0]:
        case DS.LSH:
            index_solutions = []
            for q in queries:
                time_start = time.perf_counter()
                index_solutions.append(index.get_nns_by_vector(q, k_i, search_k= 7000))
                time_end = time.perf_counter()
                dt = round(time_end - time_start, 7)
                #print(f'\tQuerying LSH Index: \t{time_end - time_start:.3f} seconds')
                timings.append({
                    'instance' : i,
                    'algo' : to_string(ds),
                    'event' : 'query',
                    'dt' : dt
                })
            index_solutions = np.asarray(index_solutions)
            index_solutions.astype(np.intp, copy=False)
        case DS.HNSW:
            index.set_ef(700)
            index_solutions = []
            for q in queries:
                time_start = time.perf_counter()
                solutions, _ = index.knn_query(q, k_i)
                time_end = time.perf_counter()
                dt = round(time_end - time_start, 7)
                timings.append({
                        'instance' : i,
                        'algo' : to_string(ds),
                        'event' : 'query',
                        'k' : k_i,
                        'dt' : dt
                })
                index_solutions.append(solutions[0])
                #print(f'\tQuerying HNSW Index: \t{time_end - time_start:.3f} seconds')
        case DS.KD:
            index_solutions = []
            for q in queries:
                q = q.reshape(1, -1)
                time_start = time.perf_counter()
                solutions = rand_samples[index.query(q, k=k_i, return_distance=False)]
                time_end = time.perf_counter()
                dt = round(time_end - time_start, 7)
                index_solutions.append(solutions[0])
                timings.append({
                    'instance' : i,
                    'algo' : to_string(ds),
                    'event' : 'query',
                    'dt' : dt
                })
                #print(f'\tQuerying KD-Tree@{ds[1]} Index: \t{time_end - time_start:.3f} seconds')
        case DS.BT:
            index_solutions = []
            for q in queries:
                q = q.reshape(1, -1)
                time_start = time.perf_counter()
                solutions = rand_samples[index.query(q, k=k_i, return_distance=False)]
                time_end = time.perf_counter()
                dt = round(time_end - time_start, 7)
                index_solutions.append(solutions[0])
                #print(f'\tQuerying Ball-Tree@{ds[1]} Index: \t{time_end - time_start:.3f} seconds')
                timings.append({
                    'instance' : i,
                    'algo' : to_string(ds),
                    'event' : 'query',
                    'dt' : dt
                })
        case DS.RS:
            index_solutions = []
            for _ in range(len(queries)):
                time_start = time.perf_counter()
                index_solutions.append(np.random.choice(len(sites), k_i, replace=False))
                time_end = time.perf_counter()
                dt = round(time_end - time_start, 7)
                #print(f'\tQuerying Ball-Tree@{ds[1]} Index: \t{time_end - time_start:.3f} seconds')
                timings.append({
                    'instance' : i,
                    'algo' : to_string(ds),
                    'event' : 'query',
                    'dt' : dt
                })
        case DS.BRUTE:
            index_solutions = []
            for q in queries:
                time_start = time.perf_counter()
                solutions, _ = index.knn_query(q, k_i)
                time_end = time.perf_counter()
                dt = round(time_end - time_start, 7)
                index_solutions.append(solutions[0])
                #print(f'\tQuerying Ball-Tree@{ds[1]} Index: \t{time_end - time_start:.3f} seconds')
                timings.append({
                    'instance' : i,
                    'algo' : to_string(ds),
                    'event' : 'query',
                    'dt' : dt
                })
    return index_solutions


var_name = file.attrs['var_name']

for i in range(n_instances):
    var_value = file.attrs['var_values'][i]

    print(f'Extracting instance {i}.')
    time_start = time.perf_counter()
    dataset = file['instance_' + str(i)]
    print('n_planes: ', n_planes)
    sites = dataset[()]
    site_to_distance = file['distance_' + str(i)][()]
    site_to_rank = file['ranks_' + str(i)][()]
    queries = file['queries_' + str(i)][()]
    solution = file['solution_' + str(i)][()]
    time_end = time.perf_counter()
    dt = round(time_end - time_start, 7)
    print(f'\tExtracting data: \t{dt:.3f} seconds')

    for ds in ds_to_test:
        (index, rand_sample) = create_index(ds, sites, n_dims[i], n_planes[i])

        for k_i in k:
            # Have less than k_i overall sites => skip
            if len(sites) * ds[1] <= k_i:
                continue
            index_solutions = search_index(index, ds, rand_sample, queries, k_i, i)
            for sample_idx in range(sample_size):
                solution_distances = site_to_distance[sample_idx][solution[sample_idx][:k_i]]
                sample_distances = sorted(site_to_distance[sample_idx][index_solutions[sample_idx][:k_i]])
                sample_qualities = solution_distances / sample_distances
                for q in sample_qualities:
                    qualities.append({
                        'instance' : i,
                        var_name : var_value,
                        'algo' : to_string(ds),
                        #nan occurs for division by zero, in which case the distance
                        # to the solution is zero and the quality must be one.
                        'Quality' : 1.0 if np.isnan(q) else q
                    })

            for sample_idx, neighbors in enumerate(index_solutions):
                    for r in site_to_rank[sample_idx][neighbors]:
                            ranks.append({
                                    'instance' : i,
                                    var_name : var_value,
                                    'algo' : to_string(ds),
                                    'Rank' : r
                            })


            for sample_idx, neighbors in enumerate(index_solutions):
                    neighbor_count = 0
                    for r in site_to_rank[sample_idx][neighbors]:
                            if r <= k_i:
                                    neighbor_count += 1
                    recalls.append({
                            'instance' : i,
                            var_name : var_value,
                            'algo' : to_string(ds),
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
