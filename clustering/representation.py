import numpy as np
import math

def binary_vectorizer(token_set, space,documents,cluster):
    vector = np.zeros(len(space))
    for token in token_set:
        if token in space:
            vector[space.index(token)] = 1.0
    return vector

def frequency_vectorizer(token_set, space, documents,cluster):
    vector = np.zeros(len(space))
    for token in token_set:
        if token in space:
            #import ipdb;ipdb.set_trace()
            vector[space.index(token)] = token_set.count(token) #cambie esta linea por los 1.0
    #import ipdb
    #ipdb.set_trace()       
    return vector

def count_token(token_set, space, documents,cluster):
    vector = np.zeros(len(space))
    doc_t=0   
    for token in token_set:        
        if token in space:            
            for document_name in documents[cluster]:
                document= documents[cluster][document_name]
                if token in document: 
                    doc_t+=1               
            #import ipdb;ipdb.set_trace()    
            vector[space.index(token)] = (token_set.count(token)/len(token_set))*(math.log((10/doc_t)))
    #import ipdb
    #ipdb.set_trace()       
    return vector    
