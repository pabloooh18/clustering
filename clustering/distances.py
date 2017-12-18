import numpy as np
import functools
from error_metrics import f1_class

f1=functools.partial(f1_class, cls=1.)

def matching_distance(X, Y):
    # Ensure binary vectors
    X = (X>0) * 1.
    Y = (Y>0) * 1.
    return 1 - np.sum(np.multiply(X,Y))/len(X)

def f1_distance(X, Y):
    X = (X>0) * 1.
    Y = (Y>0) * 1.
    return 1 - f1(X, Y)
