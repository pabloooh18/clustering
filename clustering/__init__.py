import pickle
import kmeans

import processing
import numpy as np

import error_metrics as err

# load precalculated docs
with open("../outputs/original/rouge_docs.p", "rb") as fin:
    docs = pickle.load(fin)
with open("../outputs/original/rouge_vect_docs.p", "rb") as fin:
    vect_docs = pickle.load(fin)
with open("../outputs/original/rouge_space.p", "rb") as fin:
    space = pickle.load(fin)
with open("../outputs/original/rouge_centroids.p", "rb") as fin:
    ideal_centroids = pickle.load(fin)

data_folder_summ10 = "../res/summ10"
data_folder_summ100 = "../res/summ100"

# get seed centroids
init_centroids = processing.get_initial_centroids(vect_docs, ideal_centroids)

base_init, means_init, tags_init, expected_init = kmeans.kmeans(
                                                     vect_docs,
                                                     init_centroids)
base_ideal, means_ideal, tags_ideal, expected_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
import ipdb;ipdb.set_trace()
del vect_docs
del docs

docs = duc.get_rouge_summary_clusters(data_folder_summ10)
vect_docs = duc.convert_to_vectors(docs, space)

summ10_init, means_init, tags_init, expected_init = kmeans.kmeans(
                                                                vect_docs,
                                                                init_centroids)
summ10_ideal, means_ideal, tags_ideal, expected_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))

import ipdb;ipdb.set_trace()
del vect_docs
del docs

docs = duc.get_rouge_summary_clusters(data_folder_summ100)
vect_docs = duc.convert_to_vectors(docs, space)

summ100_init, means_init, tags_init, expected_init = kmeans.kmeans(
                                                                vect_docs,
                                                                init_centroids)
summ100_ideal, means_ideal, tags_ideal, expected_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
