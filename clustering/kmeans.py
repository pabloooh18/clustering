from nltk.cluster import util
from sklearn.cluster import k_means_
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
import numpy as np
import inspect

def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return util.cosine_distance(X,Y)


def new_multiply(X, Y=None, Y_norm_squared=None, squared=False):
    import ipdb; ipdb.set_trace()
    return np.sum(np.multiply(X,Y))

k_means_.euclidean_distances = new_multiply




def kmeans(duc_vectorized_documents, means=np.array([])):

    tags = []
    y = []
    x = [] 
    resultado=[]      
    #means=means.tolist() #convierte en una lista pero solo una.
    #means=[list(f) for f in means] #con list() queda como una matris pero con el formato array([])  
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
            dist = util.euclidean_distance(vector, mean)
            if best_distance is None or dist < best_distance:
                best_index, best_distance = index, dist
        resultado.append(best_index)        
        #import ipdb
        #ipdb.set_trace()            
           
    # #if len(means)>0:
    #import ipdb
    #ipdb.set_trace()
    # x=[np.array(f) for f in x]
            
    # kmean = KMeansClusterer(label, distance=euclidean_distance,initial_means=None,avoid_empty_clusters=True)
    # kmeans = kmean.cluster(x,assign_clusters = True)
    #     # KMeans(n_clusters=label,init=means,algorithm="full").fit(x)
    # #else:    
    #   # kmean = KMeansClusterer(label, distance=euclidean_distance)
    #    #kmeans = kmean.cluster(x)
    #     #KMeans(n_clusters=label,algorithm="full").fit(x)

    return resultado, means, np.array(tags), np.array(y)
