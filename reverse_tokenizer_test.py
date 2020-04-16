# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:55:33 2020

@author: pmzha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:22:06 2020

@author: pmzha
"""
import functions as f
from tensorflow.python.keras import backend as k
import csv
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax

# In[] 
'''convert_to_dt'''
#source_path= "D:/USC/Lab/ChamSim/data/654.out"
#f.convert_to_dt(source_path)

'''read_dt'''
source_path= "D:/USC/Lab/ChamSim/data/dt.csv"
#source_path= "./dt.csv"
dataset_dt = pd.read_csv(source_path, header=None, index_col=None)
dataset_dt_ls=list(dataset_dt[0])[0:20]

encoded_raw_diff_str = ["%s%d" % ("1x" if x < 0 else "0x", abs(x)) for x in dataset_dt_ls]
#dataset_dt_ls = pd.DataFrame(encoded_raw_diff_str).astype(str)
                
'''tokenize'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(encoded_raw_diff_str)
encoded_final = tokenizer.texts_to_sequences([' '.join(encoded_raw_diff_str)])[0]
final_vocab_size = len(tokenizer.word_index) + 1
'''input sequence window'''
look_back = 1
sequences = f.create_windowed_dataset(encoded_final, look_back)

'''Training data preprocessing'''
X, y = sequences[0:, :-1], sequences[0:, -1]
y = y.reshape(len(y), 1)

'''binay'''
y_binay=f.convert_to_binary(data=y)
print(X.shape,y.shape)

'''split'''
'''
test_ratio=0.2
X_train0, X_test = train_test_split(X, test_size=test_ratio, shuffle=False)
y_train0, y_test = train_test_split(y_binay, test_size=test_ratio, shuffle=False)


test_ratio=0.25
X_train, X_val = train_test_split(X_train0, test_size=test_ratio, shuffle=False)
y_train, y_val = train_test_split(y_train0, test_size=test_ratio, shuffle=False)

print("X shape:",X_train.shape,X_val.shape,X_test.shape)
print("y shape:",y_train.shape,y_val.shape,y_test.shape)
'''
# In[]
# using the model
# need to save tokenizer
import functions as f
#y_pred = model_.predict([[1,1,1,1,1]])
#y_pred = model_.predict(X_test)
#y_pred[y_pred >= 0.5] = 1
#y_pred[y_pred < 0.5] = 0
#aaaaa = np.packbits(np.array(y_binay, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
#bbbbb = np.packbits(np.array(y_pred, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
aaaaa=f.convert_binary_to_dec(y_binay)
y1,y2=f.token_back(aaaaa,aaaaa,tokenizer)
'''
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
# Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)


# Creating texts 
dummy_word = "0xffffffff"    
original_testing_diffs = list(map(sequence_to_text, [aaaaa]))
original_predictions_diffs=original_testing_diffs
tmp = [((-1 if int(k[0]) == 1 else 1)*int(k[2:]), (-1 if int(l[0]) == 1 else 1)*int(l[2:])) if l is not None and k is not None and l != dummy_word and k != dummy_word else (None, None) for k,l in zip(original_testing_diffs[0], original_predictions_diffs[0])]
original_testing_diffs, original_predictions_diffs = zip(*tmp)
#original_predictions_diffs = list(map(sequence_to_text, [bbbbb]))
print(dataset_dt_ls)
print(original_testing_diffs)
'''
print(dataset_dt_ls)
print(original_testing_diffs)