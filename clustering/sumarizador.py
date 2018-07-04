import copy
import duc
import os
import sys
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from processing import *
from lcs import *
from string import punctuation

sys.setrecursionlimit(2000)
data_folder_original = "../res/original"



cluster_ids = os.listdir(data_folder_original)
documents = {}
contains_words1=[]
contains_words2=[]
contains_words3=[]
contains_words4=[]
score_doc1=[]
score_doc2=[]
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
		#sent_tokenize_list = sent_tokenize(lemmatized)
		table = str.maketrans(dict.fromkeys(punctuation))
		sent_tokenize_list = sent_tokenize(lemmatized)
		en_stopwords = set(stopwords.words('english'))
		tokens = ' '.join([word for word in word_tokenize(lemmatized.translate(table))
			  if word not in en_stopwords])    
		if var==0:
			first_document=tokens
			first_sent_tokenize_list=sent_tokenize_list
			var=1 
		else:
			#import ipdb;ipdb.set_trace() 
			tokenized_phrase1 = first_document.split(" ")
			tokenized_phrase2 = tokens.split(" ")
			
			first_document=lcs(tokenized_phrase1, tokenized_phrase2, " ")			
			#para el primer documento
			for sentence in first_sent_tokenize_list: 
				sentence = ' '.join([word for word in word_tokenize(sentence.translate(table))
			  						if word not in en_stopwords]) 
				sentence=sentence.split(" ")				
				for elem in first_document:					
					if tokenized_phrase1[elem] in sentence:
						contains_words1.append(tokenized_phrase1[elem])
				#	import ipdb;ipdb.set_trace()	
				contains_words2.append(contains_words1)	#guarda las las palabras de la lcs que salen por oracion
				score_doc1.append(len(contains_words1)/len(sentence)) # guarda los scores
				contains_words1=[]						
			#para el segundo documento						
			for sentence in sent_tokenize_list: 
				sentence = ' '.join([word for word in word_tokenize(sentence.translate(table))
			  						if word not in en_stopwords]) 
				sentence=sentence.split(" ")				
				for elem in first_document:										
					if tokenized_phrase2[elem] in sentence:
						contains_words3.append(tokenized_phrase2[elem])
							
				score_doc2.append(len(contains_words3)/len(sentence))	#guarda los scores			
				contains_words4.append(contains_words3) #guarda las las palabras de la lcs que salen por oracion
				contains_words3=[]	
			#import ipdb;ipdb.set_trace()
			
					

								
			print("LCS:", first_document)
			   




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