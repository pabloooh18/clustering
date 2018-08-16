import copy
import duc
import os
import sys
import pickle
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from processing import *
from string import punctuation
from sklearn.cluster import KMeans, MiniBatchKMeans
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import error_metrics as err

sys.setrecursionlimit(5000)
#data_folder_original = "../res/original"
#docs=[]
i=0

with open("../outputs/original/rouge_docs.p", "rb") as fin:
	data_folder_original = pickle.load(fin)

def get_rouge_document(data_folder_original):
	'''returns documents as rouge tokens, ordered by cluster'''
	documents = {}
	y=[]
	label=0
	for cluster in data_folder_original:
		#documents[cluster] = {}
		for document_name in data_folder_original[cluster]:
			y.append(label)
			document = data_folder_original[cluster][document_name]
			documents[document_name] = document
		label+=1	
	return documents, np.array(y)


docs, expected=get_rouge_document(data_folder_original)
stopset = set(stopwords.words('english'))
stopset.update(['lt','p','/p','\n','br','amp','quot','field','font','normal','span','0px','rgb','style','51', 
				'spacing','text','helvetica','size','family', 'space', 'arial', 'height', 'indent', 'letter'
				'line','none','sans','serif','transform','line','variant','weight','times', 'new','strong', 'video', 'title'
				'white','word','letter', 'roman','0pt','16','color','12','14','21', 'neue', 'apple', 'class',  ])

#labels = docs.target
vectorizer = TfidfVectorizer(stop_words=stopset,use_idf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(docs)
#import ipdb;ipdb.set_trace()

svd = TruncatedSVD(n_components=500)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

explained_variance = svd.explained_variance_ratio_.sum()
   
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
#lsa = TruncatedSVD(n_components=50, n_iter=100)

#X=lsa.fit(X)

minikm = MiniBatchKMeans(n_clusters=50, init='k-means++', n_init=1,
						 init_size=1000, batch_size=1000).fit(X)

km = KMeans(n_clusters=50, init='k-means++', max_iter=100, n_init=1
				).fit(X)

print("\n%s\n" % minikm)
print("purity : %s" % err.purity(minikm.labels_, expected))
print("purity corr : %s" % err.purity_corrected(minikm.labels_, expected))
print("f1 : %s" % err.general_f1(minikm.labels_, expected))
print("entropy : %s" % err.total_entropy(minikm.labels_, expected))


