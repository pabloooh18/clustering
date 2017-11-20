import pickle
import kmeans

import processing
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
    ideal_centroids = pickle.load(fin)

data_folder_summ10 = "../res/summ10"
data_folder_summ100 = "../res/summ100"

# get seed centroids
init_centroids = processing.get_initial_centroids(vect_docs, ideal_centroids)

base_init, means_init, tags_init, expected_init = kmeans.kmeans(
                                                     vect_docs,
                                                     init_centroids)
print("init purity : %s" % err.purity(base_init, expected_init))
print("init purity corr : %s" % err.purity_corrected(base_init, expected_init))
print("init f1 : %s" % err.general_f1(base_init, expected_init))
print("init entropy : %s" % err.total_entropy(base_init, expected_init))

base_ideal, means_ideal, tags_ideal, expected_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
print("ideal purity : %s" % err.purity(base_ideal, expected_ideal))
print("ideal purity corr : %s" % err.purity_corrected(base_ideal,
                                                      expected_ideal))
print("ideal f1 : %s" % err.general_f1(base_ideal, expected_ideal))
print("ideal entropy : %s" % err.total_entropy(base_ideal, expected_ideal))

del docs
del vect_docs

docs = duc.get_rouge_summary_clusters(data_folder_summ10)
vect_docs = duc.convert_to_vectors(docs, space)

print("\nSum10")

summ10_init, means_init, tags_init, expected_init = kmeans.kmeans(
                                                                vect_docs,
                                                                init_centroids)
print("init purity : %s" % err.purity(summ10_init, expected_init))
print("init purity corr : %s" % err.purity_corrected(summ10_init, expected_init))
print("init f1 : %s" % err.general_f1(summ10_init, expected_init))
print("init entropy : %s" % err.total_entropy(summ10_init, expected_init))

summ10_ideal, means_ideal, tags_ideal, expected_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
print("ideal purity : %s" % err.purity(summ10_ideal, expected_ideal))
print("ideal purity corr : %s" % err.purity_corrected(summ10_ideal,
                                                      expected_ideal))
print("ideal f1 : %s" % err.general_f1(summ10_ideal, expected_ideal))
print("ideal entropy : %s" % err.total_entropy(summ10_ideal, expected_ideal))

del docs
del vect_docs

docs = duc.get_rouge_summary_clusters(data_folder_summ100)
vect_docs = duc.convert_to_vectors(docs, space)

print("\nSum100")

summ100_init, means_init, tags_init, expected_init = kmeans.kmeans(
                                                                vect_docs,
                                                                init_centroids)
print("init purity : %s" % err.purity(summ100_init, expected_init))
print("init purity corr : %s" % err.purity_corrected(summ100_init, expected_init))
print("init f1 : %s" % err.general_f1(summ100_init, expected_init))
print("init entropy : %s" % err.total_entropy(summ100_init, expected_init))

summ100_ideal, means_ideal, tags_ideal, expected_ideal = kmeans.kmeans(
                                    vect_docs,
                                    np.array(list(ideal_centroids.values())))
print("ideal purity : %s" % err.purity(summ100_ideal, expected_ideal))
print("ideal purity corr : %s" % err.purity_corrected(summ100_ideal,
                                                      expected_ideal))
print("ideal f1 : %s" % err.general_f1(summ100_ideal, expected_ideal))
print("ideal entropy : %s" % err.total_entropy(summ100_ideal, expected_ideal))
