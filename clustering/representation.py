import numpy as np

def binary_vectorizer(token_set, space):
    vector = np.zeros(len(space))
    for token in token_set:
        if token in space:
            vector[space.index(token)] = 1.0
    return vector

def frequency_vectorizer(token_set, space):
    vector = np.zeros(len(space))
    for token in token_set:
        if token in space:
            #import ipdb;ipdb.set_trace()
            vector[space.index(token)] = token_set.count(token) #cambie esta linea por los 1.0
    #import ipdb
    #ipdb.set_trace()       
    return vector
