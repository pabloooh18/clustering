from nltk.cluster import util
from sklearn.cluster import k_means_
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
import numpy as np
import inspect


def kmeans_sklearn(duc_vectorized_documents, means=np.array([])):
    tags = []
    y = []
    x = np.array([])
    label = 0
    for cluster in duc_vectorized_documents:
        for document in duc_vectorized_documents[cluster]:
            y.append(label)
            tags.append(cluster + "_" + document)
            if len(x) == 0:
                x = duc_vectorized_documents[cluster][document]
            else:
                x = np.vstack([x, duc_vectorized_documents[cluster][document]])
        label += 1
    if len(means)>0:
        kmeans = KMeans(n_clusters=label,
                        init=means,
                        algorithm="full").fit(x)
    else:
        kmeans = KMeans(n_clusters=label,
                        algorithm="full").fit(x)
    return kmeans.labels_, kmeans.cluster_centers_, np.array(tags), np.array(y)


def assign_to_centroid(duc_vectorized_documents, means, distance):
    tags = []
    y = []
    x = []
    resultado=[]
    label = 0
    for cluster in duc_vectorized_documents:
        for document in duc_vectorized_documents[cluster]:
            y.append(label)
            tags.append(cluster + "_" + document)
            if len(x) == 0:
                x = duc_vectorized_documents[cluster][document]
            else:
                x = np.vstack([x, duc_vectorized_documents[cluster][document]])
        label += 1

    for vector in x:
        best_distance = best_index = None
        for index in range(len(means)):
            mean = means[index]
            dist = distance(vector, mean)
            if best_distance is None or dist <= best_distance:
                best_index, best_distance = index, dist
        resultado.append(best_index)

    return resultado, means, np.array(tags), np.array(y)
