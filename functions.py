# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:56:18 2020

@author: pmzha
"""

import csv
import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

def convert_to_dt(source_path):
    dataset = pd.read_csv(source_path, header=None, index_col=None)

    '''conver to deltas'''
    dataset_dt=dataset.diff()[1:]
    dataset_dt.to_csv("dt.csv", index=None)
    
def create_windowed_dataset(data, look_back):
    """
    Create the dataset by grouping windows of memory accesses together (using the look_back parameter)

    data: it should be a list of integers
    """
    sequences = list()
    for i in range(look_back, len(data)):
        sequence = data[i - look_back:i + 1]
        sequences.append(sequence)
    return np.array(sequences)

def difference(dataset, interval=1):
    """
    Calculates the difference between a time-series and a lagged version of it
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def convert_to_binary(data,bit_size=16):
    data_binary = np.array([[int(d) for i in j for d in str('{0:016b}'.format(i))] for j in list(data)])
    return data_binary     