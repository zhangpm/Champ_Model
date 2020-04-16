# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:19:53 2020

@author: pmzha
"""
from tensorflow.keras.models import load_model
import pickle
import numpy as np

def convert_binary_to_dec(data,bit_size=16):
    #return np.reshape(np.array(list([int("".join(str(i) for i in List),2) for List in data])),(-1,1))
    dec = np.packbits(np.array(data, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
    return dec
    # return data_dec 

def token_back(y_pred_dec,y_test_dec,tokenizer)  :
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return(words)
    dummy_word = "0xffffffff"    
    original_testing_diffs = list(map(sequence_to_text, [y_pred_dec]))
    original_predictions_diffs = list(map(sequence_to_text, [y_test_dec]))
    tmp = [((-1 if int(k[0]) == 1 else 1)*int(k[2:]), (-1 if int(l[0]) == 1 else 1)*int(l[2:])) if l is not None and k is not None and l != dummy_word and k != dummy_word else (None, None) for k,l in zip(original_testing_diffs[0], original_predictions_diffs[0])]
    original_testing_diffs, original_predictions_diffs = zip(*tmp)   
    return list(original_testing_diffs), list(original_predictions_diffs)


def call_model_pred(X_test,model_dir='my_model.h5',tokenizer='tokenizer.pickle'):
    model_file_name=model_dir
    model_ = load_model(model_file_name)
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    X_test2=np.array([X_test])
    y_pred1 = model_.predict(X_test2)
    y_pred1[y_pred1 >= 0.5] = 1
    y_pred1[y_pred1 < 0.5] = 0
    
    b1 = convert_binary_to_dec(y_pred1)
    diff,nomean=token_back(b1,b1,tokenizer)
    return diff[0]