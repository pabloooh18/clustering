import numpy as np
from collections import OrderedDict

def mse(predicted, observed):
    return np.sum((predicted - observed)**2) / len(observed)

def confusion_matrix(predicted, observed):
    classes = set(observed.flatten())
    matrix = OrderedDict({})
    for cls in classes:
        matrix[cls] = OrderedDict(((cls2, 0) for cls2 in classes))

    for cls in classes:
        matrix[cls][cls] = np.sum(np.logical_and((predicted == cls),
                                                 (observed  == cls)
                                                )
                                 )
        for cls2 in classes - {cls}:
            matrix[cls2][cls] = np.sum(np.logical_and((predicted == cls2),
                                                      (observed  == cls)
                                                     )
                                      )
    return matrix

def print_confusion_matrix(matrix):
    classes = matrix.keys()
    row = ""
    for cls in classes:
        row += "\t" + cls
    print(row)
    for i in classes:
        row = i
        for j in classes:
            row += "\t" + str(matrix[i][j])
        print(row)

def precision(predicted, observed, cls):
    true_positives = np.sum(np.logical_and((predicted == cls),
                                           (observed == cls)
                                          )
                           )
    return true_positives/len(observed)

def recall(predicted, observed, cls):
    true_positives = np.sum(np.logical_and((predicted == cls),
                                           (observed == cls)
                                          )
                           )
    false_negatives =
    return true_positives/(true_positives + false_negatives)

def f1(predicted, observed, cls):
    prec = precision(predicted, observed, cls)
    rec = recall(predicted, observed, cls)
    return 2*((prec*rec)/(prec+rec))

# observed = np.array(("cat", "cat", "cat", "cat", "dog", "dog", "dog", "human", "human", "donkey"))
# predicted = np.array(("dog", "cat", "cat", "dog", "donkey", "human", "human", "human", "donkey", "dog"))
# mat = confusion_matrix(predicted, observed)
# print_confusion_matrix(mat)
# print(list(enumerate(list(observed))))
# print(list(enumerate(list(predicted))))
