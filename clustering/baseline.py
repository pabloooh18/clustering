import pickle
import kmeans
from nltk.cluster import util

import numpy as np

with open("../outputs/original/rouge_docs.p", "rb") as fin:
    docs = pickle.load(fin)
with open("../outputs/original/rouge_vect_docs.p", "rb") as fin:
    vect_docs = pickle.load(fin)
with open("../outputs/original/rouge_space.p", "rb") as fin:
    space = pickle.load(fin)
with open("../outputs/original/rouge_centroids.p", "rb") as fin:
    ideal_centroids = pickle.load(fin)

# get seed centroids
init_centroids = np.array([])
for cluster in vect_docs:
    candidate = None
    min_distance = None
    for document in vect_docs[cluster]:
        if min_distance == None:
            min_distance = util.cosine_distance(
                                            ideal_centroids[cluster],
                                            vect_docs[cluster][document])
            candidate = vect_docs[cluster][document]
        else:
            candidate_distance = util.cosine_distance(
                                            ideal_centroids[cluster],
                                            vect_docs[cluster][document])
            if candidate_distance < min_distance:
                min_distance = candidate_distance
                candidate = vect_docs[cluster][document]
    if len(init_centroids) == 0:
        init_centroids = candidate
    else:
        init_centroids = np.vstack([init_centroids, candidate])
baseline_init, result_init, tags_init, means_init = kmeans.kmeans(
                                                                vect_docs,
                                                                init_centroids)
baseline_ideal, result_ideal, tags_ideal, means_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
baseline_ideal, result_ideal, tags_ideal, means_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
import ipdb;ipdb.set_trace()
