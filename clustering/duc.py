import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import numpy as np
import pickle

import os
import json

from processing import *

def get_rouge_tokens(original_text):
    '''lemmatized tokens with no stopwords nor punctuation
       Frequency irrelevant
       Capitalized named entities
    '''
    #import ipdb;ipdb.set_trace()
    lemmatized = lemmatize(original_text)
    table = str.maketrans(dict.fromkeys(punctuation))
    en_stopwords = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(lemmatized.translate(table))
              if word not in en_stopwords]
    return set(tokens)

def get_rouge_tokens_original(original_text):
    '''lemmatized tokens with no stopwords nor punctuation
       Frequency relevant
       Capitalized named entities
    '''
    #import ipdb;ipdb.set_trace()
    lemmatized = lemmatize(original_text)
    table = str.maketrans(dict.fromkeys(punctuation))
    en_stopwords = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(lemmatized.translate(table))
              if word not in en_stopwords]
    return tokens

def get_rouge_document_clusters(data_folder_original):
    '''returns documents as rouge tokens, ordered by cluster'''
    cluster_ids = os.listdir(data_folder_original)
    documents = {}

    # Gets the tokens
    for cluster in cluster_ids:
        cluster_folder = data_folder_original + "/" + cluster
        documents[cluster] = {}
        for document in os.listdir(cluster_folder):
            document_path = cluster_folder + "/" + document
            tree = ET.parse(document_path)
            root = tree.getroot()
            original_text = root.find("TEXT").text
            token_set = get_rouge_tokens(original_text) # para usar la repeticion, usar get_rouge_tokens_original
            documents[cluster][document] = token_set
    return documents

def get_rouge_summary_clusters(data_folder_original):
    '''returns documents as rouge tokens, ordered by cluster'''
    cluster_ids = os.listdir(data_folder_original)
    documents = {}

    # Gets the tokens
    for cluster in cluster_ids:
        cluster_folder = data_folder_original + "/" + cluster
        documents[cluster] = {}
        for document in os.listdir(cluster_folder):
            document_path = cluster_folder + "/" + document
            with open(document_path, "r") as fin:
                original_text = fin.read()
            token_set = get_rouge_tokens(original_text) # para usar la repeticion, usar get_rouge_tokens_original
            documents[cluster][document] = token_set
    return documents

def convert_to_vectors(documents, vector_space, vectorizer):
    # Calculates the representation
    vectorized_documents = {}
    for cluster in documents:
        vectorized_documents[cluster] = {}
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            #import ipdb;ipdb.set_trace()
            #if document_name=='APW19981027.0491': import ipdb;ipdb.set_trace()
            vectorized_documents[cluster][document_name] = vectorizer(
                                                                document,
                                                                vector_space,documents,cluster)
    return vectorized_documents

def get_vector_space_from_clusters(documents):
    token_sets = []
    for cluster in documents.keys():
        for token_set in documents[cluster].values():
            token_sets.append(token_set)

    # Calculates the vector space
    space = get_space(token_sets)
    return space

def get_cluster_centroids(vectorized_documents):
    centroids = {}
    for cluster in vectorized_documents:
        centroids[cluster] = get_centroid(tuple(
                                        vectorized_documents[cluster].values()
                                            )
                                )
    return centroids

def get_matching_vectors(vectorized_documents):
    '''Vectors for the multiplication-based distance
       logical and between all of them.
    '''
    centroids = {}
    for cluster in vectorized_documents:
        for docid, vector in vectorized_documents[cluster].items():
            try:
                centroids[cluster] = np.add(centroids[cluster],
                                            vector)
            except:
                #print( "%s (%s)" % (cluster, docid))
                centroids[cluster] = vector
                continue
        # To binary
        centroids[cluster] = (centroids[cluster]>0)*1.
    return centroids


def jsonify(dictionary):
    if isinstance(dictionary, np.ndarray) or\
       isinstance(dictionary, set):
        return list(dictionary)
    elif isinstance(dictionary, dict):
        for key in dictionary.keys():
            dictionary[key] = jsonify(dictionary[key])
    return dictionary

def dump(dictionary, output_path):
    with open(output_path, "w") as out:
        out.write(json.dumps(jsonify(dictionary)))

def pickle_dump(dictionary, output_path):
    with open(output_path, "wb") as out:
        pickle.dump(dictionary, out)

def pickle_dumps(dictionary, output_path):
    n_bytes = 2**11
    bytes_out = pickle.dumps(dictionary)
    max_bytes = len(bytes_out)-1
    with open(output_path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            f_out.flush()
