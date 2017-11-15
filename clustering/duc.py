import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import os

from processing import *

def get_rouge_tokens(original_text):
    '''lemmatized tokens with no stopwords nor punctuation
       Frequency irrelevant
       Capitalized named entities
    '''
    lemmatized = lemmatize(original_text)
    table = str.maketrans(dict.fromkeys(punctuation))
    en_stopwords = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(lemmatized.translate(table))
              if word not in en_stopwords]
    return set(tokens)

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
            token_set = get_rouge_tokens(original_text)
            documents[cluster][document] = token_set
        break
    return documents


def convert_to_vectors(documents, vector_space):
    # Calculates the representation
    vectorized_documents = {}
    for cluster in documents:
        vectorized_documents[cluster] = {}
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            vectorized_documents[cluster][document_name] = \
                                                    get_vector_representation(
                                                                    document,
                                                                    vector_space)
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
