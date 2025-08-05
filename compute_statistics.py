#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import h5py

if(len(sys.argv) <= 1):
    raise Exception("Please supply a filepath for data to plot!")

filepath, _ = os.path.splitext(sys.argv[1])

qualities = pd.read_hdf(filepath + "_results.h5", key="qualities")

var_name = h5py.File(filepath + "_results.h5", 'r').attrs['var_name']

deviations = qualities.groupby([var_name, 'algo'])['Quality'].std()

mins = qualities.groupby([var_name, 'algo'])['Quality'].min()

means = qualities.groupby([var_name, 'algo'])['Quality'].mean()

maxs = qualities.groupby([var_name, 'algo'])['Quality'].max()

print(deviations)
print(mins)
print(means)
print(maxs)
