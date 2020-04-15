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


souce_path= "D:/USC/Lab/ChamSim/data/654.out"
dataset = pd.read_csv(souce_path, header=None, index_col=None)
dataset_ls=list(dataset[0].astype(str))
'''conver to deltas'''
#dataset_dt=dataset.diff()[1:]
#dataset_dt.to_csv("dt.csv", index=None)

#dataset_dt.apply(pd.value_counts)
'''sample total lines'''
#max_sample_size=dataset_dt.count()[0]

'''tokenize'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset_ls)
encoded_final = tokenizer.texts_to_sequences([' '.join(dataset_ls)])[0]
final_vocab_size = len(tokenizer.word_index) + 1

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


