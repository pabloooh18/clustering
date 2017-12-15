import numpy as np

def matching_distance(X, Y):
    # Ensure binary vectors
    X = (X>0)*.1
    Y = (Y>0)*.1
    return 1 - np.sum(np.multiply(X,Y))/len(X)

