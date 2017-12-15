import pickle
from nltk.cluster import util

from representation import binary_vectorizer
import processing
import algorithms
import distances
import duc
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
    medium_vectors = pickle.load(fin)

matching_vectors = duc.get_matching_vectors(vect_docs) # Calculated only once

data_folder_summ10 = "../res/summ10"
data_folder_summ100 = "../res/summ100"

def print_results(vectors, centroids, distance, title="Result"):
    results, means, tags, expected = algorithms.assign_to_centroid(
                                        vectors,
                                        np.array(list(centroids.values())),
                                        distance=distance
                                        )
    print("\n%s\n" % title)
    print("purity : %s" % err.purity(results, expected))
    print("purity corr : %s" % err.purity_corrected(results, expected))
    print("f1 : %s" % err.general_f1(results, expected))
    print("entropy : %s" % err.total_entropy(results, expected))


print_results(vect_docs,
              medium_vectors,
              util.euclidean_distance,
              "Original (euclidean)")

print_results(vect_docs,
              medium_vectors,
              util.cosine_distance,
              "Original (cosine)")

print_results(vect_docs,
              matching_vectors,
              distances.matching_distance,
              "Original (multiplication)")
del docs
del vect_docs


docs = duc.get_rouge_summary_clusters(data_folder_summ100)
vect_docs = duc.convert_to_vectors(docs, space, binary_vectorizer)
print_results(vect_docs,
              medium_vectors,
              util.euclidean_distance,
              "Summ100 (euclidean)")

print_results(vect_docs,
              medium_vectors,
              util.cosine_distance,
              "Summ100 (cosine)")

print_results(vect_docs,
              matching_vectors,
              distances.matching_distance,
              "Summ100 (multiplication)")
del docs
del vect_docs

import ipdb;ipdb.set_trace()

docs = duc.get_rouge_summary_clusters(data_folder_summ10)
vect_docs = duc.convert_to_vectors(docs, space, binary_vectorizer)
print_results(vect_docs,
              medium_vectors,
              util.euclidean_distance,
              "Summ10 (euclidean)")
del docs
del vect_docs
