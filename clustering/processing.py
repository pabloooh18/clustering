from nltk.tokenize import sent_tokenize
from pycorenlp import StanfordCoreNLP
import numpy as np

nlp= StanfordCoreNLP("http://localhost:9000")

def lemmatize(original_text):
    sentences = sent_tokenize(original_text)
    processed_sentences = []
    for sentence in sentences:
        output = nlp.annotate(sentence,
                              properties={
                                          'annotators': 'lemma',
                                          'outputFormat': 'json'
                              }
        )
        lemmas = []
        for token in output['sentences'][0]['tokens']:
            lemmas.append(token['lemma'])
        processed_sentences.append(" ".join(lemmas))
    return " ".join(processed_sentences)

def get_space(token_sets):
    space = set()
    for token_set in token_sets:
        space = space.union(token_set)
    return sorted(list(space))

def get_vector_representation(token_set, space):
    vector = np.zeros(len(space))
    for token in token_set:
        if token in space:
            vector[space.index(token)] = 1.0
    return vector

def get_document_vectors(documents, rouge_space):
    '''documents is a cluster of documents'''
    # Calculates the representation
    vectorized_documents = {}
    for cluster in documents:
        vectorized_documents[cluster] = {}
        for document_name in documents[cluster]:
            document = documents[cluster][document_name]
            vectorized_documents[cluster][document_name] = \
                                                    get_vector_representation(
                                                                    document,
                                                                    rouge_space)
    return vectorized_documents

def get_centroid(vector_cluster):
    return np.mean(vector_cluster, axis=0)
