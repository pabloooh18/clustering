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

def purity(predicted, observed):
    matrix = confusion_matrix(predicted, observed)
    max_values = []
    for cls in matrix.keys():
        max_values.append(max(matrix[cls].values()))
    return sum(max_values)/len(observed)

def true_positives(predicted, observed, cls):
    return np.sum(np.logical_and((predicted == cls), (observed == cls)))

def true_negatives(predicted, observed, cls):
    return np.sum(np.logical_and((predicted != cls), (observed != cls)))

def false_positives(predicted, observed, cls):
    return np.sum(np.logical_and((predicted == cls), (observed != cls)))

def false_negatives(predicted, observed, cls):
    return np.sum(np.logical_and((predicted != cls), (observed == cls)))

def table_of_confusion(predicted, observed, cls):
    classes = [cls, "not " + cls]
    table = OrderedDict({})
    for i in classes:
        table[i] = OrderedDict(((cls2, 0) for cls2 in classes))
    table[cls][cls] = true_positives(predicted, observed, cls)
    table[cls]["not " + cls] = false_positives(predicted, observed, cls)
    table["not " + cls][cls] = false_negatives(predicted, observed, cls)
    table["not " + cls]["not " + cls] = true_negatives(predicted, observed, cls)
    return table

def precision(predicted, observed, cls):
    tp = true_positives(predicted, observed, cls)
    fp = false_positives(predicted, observed, cls)
    return tp/(tp + fp)

def recall(predicted, observed, cls):
    tp = true_positives(predicted, observed, cls)
    fn = false_negatives(predicted, observed, cls)
    return tp/(tp + fn)

def f1(predicted, observed, cls):
    prec = precision(predicted, observed, cls)
    rec = recall(predicted, observed, cls)
    return 2 * ((prec * rec) / (prec + rec))

# observed = np.array(("cat", "cat", "cat", "cat", "dog", "dog", "dog", "human", "human", "donkey"))
# predicted = np.array(("dog", "cat", "cat", "dog", "donkey", "cat", "human", "human", "donkey", "dog"))
# mat = confusion_matrix(predicted, observed)
# print(purity(predicted, observed))
# table = table_of_confusion(predicted, observed, "cat")
# print_confusion_matrix(mat)
# print_confusion_matrix(table)
# print(list(enumerate(list(observed))))
# print(list(enumerate(list(predicted))))
# print("p: %0.2f" % precision(predicted, observed, "cat"))
# print("r: %0.2f" % recall(predicted, observed, "cat"))
# print("f1: %0.2f" % f1(predicted, observed, "cat"))
