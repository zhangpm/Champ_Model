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
dataset_dt_ls=list(dataset_dt[0].astype(str))

'''tokenize'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset_dt_ls)
encoded_final = tokenizer.texts_to_sequences([' '.join(dataset_dt_ls)])[0]
final_vocab_size = len(tokenizer.word_index) + 1
'''input sequence window'''
look_back = 3
sequences = f.create_windowed_dataset(encoded_final, look_back)

'''Training data preprocessing'''
X, y = sequences[40000:60000, :-1], sequences[40000:60000, -1]
y = y.reshape(len(y), 1)

'''binay'''
y_binay=f.convert_to_binary(data=y)

'''split'''
test_ratio=0.3
X_train, X_test = train_test_split(X, test_size=test_ratio, shuffle=False)
y_train, y_test = train_test_split(y_binay, test_size=test_ratio, shuffle=False)

# In[] 
from my_model import my_model
'''train model'''
embedding_dim=10
i_dim=look_back
o_dim=y_train.shape[1]
batch_size=200
num_epochs=5
model_ = my_model(final_vocab_size,batch_size, embedding_dim, i_dim, o_dim)
history=model_.train(X_train, y_train,X_test, y_test, num_epochs, batch_size)

# In[] 

y_pred = model_.predict(X_test,batch_size=batch_size)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
aaaaa = np.packbits(np.array(y_test, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
bbbbb = np.packbits(np.array(y_pred, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(np.array(aaaaa), np.array(bbbbb))
print(accuracy)