import pickle

with open("../res/rouge_docs.txt", "rb") as fin:
    docs = pickle.load(fin)
with open("../res/rouge_vect_docs.txt", "rb") as fin:
    vect_docs = pickle.load(fin)
with open("../res/rouge_space.txt", "rb") as fin:
    space = pickle.load(fin)
with open("../res/rouge_centroids.txt", "rb") as fin:
    centroids = pickle.load(fin)

maximum_terms = 0
for cluster in docs.keys():
    for document in docs[cluster]:
        maximum_terms += len(docs[cluster][document])
print(maximum_terms)
print(len(space))
import ipdb;ipdb.set_trace()
