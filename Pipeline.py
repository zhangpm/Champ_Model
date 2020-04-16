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
import pickle

# In[] 
'''convert_to_dt'''
#source_path= "D:/USC/Lab/ChamSim/data/654.out"
#f.convert_to_dt(source_path)

'''read_dt'''
source_path= "D:/USC/Lab/ChamSim/data/dt.csv"
#source_path= "./dt.csv"
dataset_dt = pd.read_csv(source_path, header=None, index_col=None)
dataset_dt_ls=list(dataset_dt[0])[0:600000]

dataset_dt_ls_x = ["%s%d" % ("1x" if x < 0 else "0x", abs(x)) for x in dataset_dt_ls]

'''tokenize'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset_dt_ls_x)
encoded_final = tokenizer.texts_to_sequences([' '.join(dataset_dt_ls_x)])[0]
final_vocab_size = len(tokenizer.word_index) + 1
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
'''input sequence window'''
look_back = 5
sequences = f.create_windowed_dataset(encoded_final, look_back)

'''Training data preprocessing'''
X, y = sequences[:, :-1], sequences[:, -1]
y = y.reshape(len(y), 1)

'''binay'''
y_binay=f.convert_to_binary(data=y)
print(X.shape,y.shape)

'''split'''
'''val'''
test_ratio=0.2
X_train0, X_test = train_test_split(X, test_size=test_ratio, shuffle=False)
y_train0, y_test = train_test_split(y_binay, test_size=test_ratio, shuffle=False)

'''test'''
test_ratio=0.25
X_train, X_val = train_test_split(X_train0, test_size=test_ratio, shuffle=False)
y_train, y_val = train_test_split(y_train0, test_size=test_ratio, shuffle=False)

print("X shape:",X_train.shape,X_val.shape,X_test.shape)
print("y shape:",y_train.shape,y_val.shape,y_test.shape)
# In[] 
from my_model import my_model
'''train model'''
model_file_name='my_model2.h5'
embedding_dim=10
i_dim=look_back
o_dim=y_train.shape[1]
batch_size=256
num_epochs=20
model_ = my_model(final_vocab_size, batch_size,embedding_dim, i_dim, o_dim)
history=model_.train(X_train, y_train,X_val, y_val, num_epochs, batch_size)
model_.model.save(model_file_name)
# In[] 
import matplotlib.pyplot as plt
fig1 = plt.figure(dpi=50, figsize=(10, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel("Epoch")
#plt.ylabel("Training loss")
plt.legend(loc="best")
fig2 = plt.figure(dpi=50, figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuray')
plt.xlabel("Epoch")
plt.legend(loc="best")
# In[]
'''
y_pred = model_.predict(X_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
aaaaa = np.packbits(np.array(y_test, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
bbbbb = np.packbits(np.array(y_pred, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(np.array(aaaaa), np.array(bbbbb))
print(accuracy)
'''
# In[]
#evaluate
# need to save tokenizer

#y_pred = model_.predict([[1,1,1,1,1]])
#X_test=[[1,1,1]]
y_pred = model_.predict(X_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
aaaaa = f.convert_binary_to_dec(y_test)
bbbbb = f.convert_binary_to_dec(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(np.array(aaaaa), np.array(bbbbb))
print("accuracy:",accuracy)

#bbbbb = np.packbits(np.array(y_pred, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
original_testing_diffs,original_predictions_diffs=f.token_back(aaaaa,bbbbb,tokenizer)

#print(original_testing_diffs,original_predictions_diffs)

# In[]
# using the model

X_test2=np.array([X_test[0]])
y_pred1 = model_.predict(X_test2)
y_pred1[y_pred1 >= 0.5] = 1
y_pred1[y_pred1 < 0.5] = 0

a1=[0]
b1 = f.convert_binary_to_dec(y_pred1)
diff,nomean=f.token_back(b1,b1,tokenizer)
print(diff)
