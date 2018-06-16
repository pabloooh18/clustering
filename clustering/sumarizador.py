import copy
import duc
import os
import sys
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from processing import *
from lcs import *
from string import punctuation

sys.setrecursionlimit(2000)
data_folder_original = "../res/original"



cluster_ids = os.listdir(data_folder_original)
documents = {}
#import ipdb;ipdb.set_trace()
# Gets the tokens
for cluster in cluster_ids:
	
	cluster_folder = data_folder_original + "/" + cluster
	documents[cluster] = {}
	var=0

	for document in os.listdir(cluster_folder):
		document_path = cluster_folder + "/" + document
		tree = ET.parse(document_path)
		root = tree.getroot()
		original_text = root.find("TEXT").text
		#import ipdb;ipdb.set_trace() 
		lemmatized = lemmatize(original_text)
		table = str.maketrans(dict.fromkeys(punctuation))
		en_stopwords = set(stopwords.words('english'))
		tokens = ' '.join([word for word in word_tokenize(lemmatized.translate(table))
			  if word not in en_stopwords])    
		if var==0:
			first_document=tokens
			var=1 
		else:
			#import ipdb;ipdb.set_trace() 
			tokenized_phrase1 = first_document.split(" ")
			tokenized_phrase2 = tokens.split(" ")
			import ipdb;ipdb.set_trace()
			first_document=lcs(tokenized_phrase1, tokenized_phrase2, " ") 
			print("LCS:", first_document)
			#import ipdb;ipdb.set_trace()   




		# if var==1:
		#     #import ipdb;ipdb.set_trace() 
		#     tokenized_phrase1 = first_document.split(" ")
		#     tokenized_phrase2 = tokens.split(" ")
		#     print("LCS:", lcs(tokenized_phrase1, tokenized_phrase2, " "))
		#     #print("All LCS:", all_lcs(first_document, tokens, " "))
		#     var=0
		# if var==0:
		#     first_document=tokens
		#     var=1   