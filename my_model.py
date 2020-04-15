# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:33:30 2020

@author: pmzha
"""

import os
import numpy as np
#from Evaluate import Evaluate
from tensorflow.python.keras import backend as k
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class my_model:
    def __init__(self, vocab_size, embedding_dim, i_dim, o_dim):
        self.i_dim = i_dim
        self.o_dim = o_dim

        self.model = Sequential()
   #     self.model.add(Embedding(vocab_size, embedding_dim, input_length=i_dim))
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=i_dim))
        self.model.add(LSTM(50))
       # self.model.add(CuDNNLSTM(50))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(20,activation='sigmoid'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(o_dim,activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model_name = 'DCLSTM'
        self.model.summary()

    def train(self,X_train, y_train, X_test, y_test, num_epochs, batch_size):
        self.model.fit(X_train,
                y_train,
                epochs=num_epochs,
                shuffle=False,
                batch_size=batch_size)
               # validation_data=(X_test, y_test))
        
    def predict(self,X):
        return self.model.predict(np.asarray(X))