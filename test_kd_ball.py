#!/usr/bin/env python3

from sklearn.neighbors import KDTree, BallTree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import time
import numpy as np
import h5py



CHUNK_SIZE = 20
file = h5py.File('data/sift1m.h5', 'r')
n_instances = file.attrs['n_instances'].astype(int)
k = [i for i in range(10, 1010, 10)]
n_sites = file.attrs['n_sites'].astype(int).tolist()
description = file.attrs['description']
hypothesis = file.attrs['hypothesis']
sample_size = file.attrs['sample_size']
n_dims = file.attrs['n_dims'].astype(int).tolist()
n_planes = file.attrs['n_planes'].astype(int).tolist()


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
kdTree = KDTree(sites, leaf_size = 30, metric='euclidean')
time_end = time.perf_counter()
dt = round(time_end - time_start, 7)
print(f'\tCreating KDTree: \t{time_end - time_start:.3f} seconds')
timings.append({
    'instance' : 0,
    'algo' : 'kd-tree',
    'event' : 'creation',
    'dt' : dt
})

time_start = time.perf_counter()
ballTree = BallTree(sites, leaf_size = 30, metric='euclidean')
time_end = time.perf_counter()
dt = round(time_end - time_start, 7)
print(f'\tCreating Ball Tree: \t{time_end - time_start:.3f} seconds')
timings.append({
    'instance' : 0,
    'algo' : 'ball-tree',
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
        kd_solutions = kdTree.query(queries, k=k_i, return_distance=False)
        time_end = time.perf_counter()
        dt = round(time_end - time_start, 7)
        print(f'\tQuerying KDTree: \t{time_end - time_start:.3f} seconds')
        timings.append({
            'instance' : 0,
            'algo' : 'kd-tree',
            'event' : 'query',
            'dt' : dt / CHUNK_SIZE
        })


        time_start = time.perf_counter()
        bt_solutions = ballTree.query(queries, k=k_i, return_distance=False)
        time_end = time.perf_counter()
        dt = round(time_end - time_start, 7)
        print(f'\tQuerying Ball Tree: \t{time_end - time_start:.3f} seconds')
        timings.append({
            'instance' : 0,
            'algo' : 'ball-tree',
            'event' : 'query',
            'dt' : dt / CHUNK_SIZE
        })


        kd_quality = []
        bt_quality = []
        for sample_idx in range(CHUNK_SIZE):
            kd_quality.append(site_to_distance[sample_idx][solution[sample_idx][:k_i]] / sorted(site_to_distance[sample_idx][kd_solutions[sample_idx][:k_i]]))
            bt_quality.append(site_to_distance[sample_idx][solution[sample_idx][:k_i]] / sorted(site_to_distance[sample_idx][bt_solutions[sample_idx][:k_i]]))

        #print(f'kd_quality {kd_quality}')
        #print(f'bt_quality {bt_quality}')

        var_name = file.attrs['var_name']
        var_value = file.attrs['var_values'][i]

        for sample in kd_quality:
            for q in sample:
                    qualities.append({
                    var_name : var_value,
                    'algo' : 'kd-tree',
                    'Quality' : q
                    })

        for sample in bt_quality:
            for q in sample:
                    qualities.append({
                    var_name : var_value,
                    'algo' : 'ball-tree',
                    'Quality' : q
                    })

        for sample_idx, neighbors in enumerate(kd_solutions):
            for r in site_to_rank[sample_idx][neighbors]:
                    ranks.append({
                            var_name : var_value,
                            'algo' : 'kd-tree',
                            'Rank' : r
                    })

        for sample_idx, neighbors in enumerate(bt_solutions):
            for r in site_to_rank[sample_idx][neighbors]:
                    ranks.append({
                            var_name : var_value,
                            'algo' : 'ball-tree',
                            'Rank' : r
                    })

        for sample_idx, neighbors in enumerate(kd_solutions):
            neighbor_count = 0
            for r in site_to_rank[sample_idx][neighbors]:
                if r <= k_i:
                    neighbor_count += 1
            recalls.append({
                    var_name : var_value,
                    'algo' : 'kd-tree',
                    'Recall' : neighbor_count / k_i,
                    'k' : k_i
            })

        for sample_idx, neighbors in enumerate(bt_solutions):
            neighbor_count = 0
            for r in site_to_rank[sample_idx][neighbors]:
                if r <= k_i:
                    neighbor_count += 1
            recalls.append({
                    var_name : var_value,
                    'algo' : 'ball-tree',
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

target_path = "data/sift1m_kd_bt"

timings.to_hdf(target_path + "_results.h5", key="timings", format="table")
qualities.to_hdf(target_path + "_results.h5", key="qualities", format="table")
ranks.to_hdf(target_path + "_results.h5", key="ranks", format="table")
recalls.to_hdf(target_path + "_results.h5", key="recalls", format="table")

results = h5py.File(target_path + "_results.h5", 'r+')

results.attrs['var_name'] = file.attrs['var_name']
