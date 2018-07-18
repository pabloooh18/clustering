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
sentence_list=[]
resume=""
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
		lemmatized = lemmatize(original_text)		
		table = str.maketrans(dict.fromkeys(punctuation))
		sent_tokenize_list = sent_tokenize(lemmatized)
		en_stopwords = set(stopwords.words('english'))
		tokens = ' '.join([word for word in word_tokenize(lemmatized.translate(table))
			  if word not in en_stopwords])
		#import ipdb;ipdb.set_trace()
		for sentence in sent_tokenize_list: #quita puntuacion y stopwords de cada oracion			
			sentence = ' '.join([word for word in word_tokenize(sentence.translate(table))
			  						if word not in en_stopwords])
			sentence_list.append(sentence)  						 
		#import ipdb;ipdb.set_trace()	      
		if var==0:
			first_document=tokens
			first_sentence_list=sentence_list
			sentence_list=[]
			var=1 
		else:
			#import ipdb;ipdb.set_trace() 
			tokenized_phrase1 = first_document.split(" ")
			tokenized_phrase2 = tokens.split(" ")
			
			# tokenized_phrase1 = "Hay una abeja Hay una flor La abeja hace miel La miel es cara".split(" ")
			# tokenized_phrase2 = "La abeja va hasta flor para hacer miel que se vende cara en el mercado".split(" ")
			# first_sentence_list = ["Hay una abeja","Hay una flor","La abeja hace miel","La miel es cara"]
			# sentence_list = ["La abeja va hasta flor para hacer miel que se vende cara en el mercado"]			
			result_lcs=lcs(tokenized_phrase1, tokenized_phrase2, " ")
			count=0				
			while count == 0:
				#import ipdb;ipdb.set_trace()
				for elem in result_lcs:					
					if tokenized_phrase1[elem] in first_sentence_list[count]:
						contains_words1.append(elem)
						
					if tokenized_phrase2[elem] in sentence_list[count]:
						contains_words3.append(elem)
			
				score_doc1=len(contains_words1)/len(first_sentence_list[count].split(" "))	
				score_doc2=len(contains_words3)/len(sentence_list[count].split(" "))
				if score_doc1 > score_doc2:
					resume= resume + first_sentence_list[count] + " "
					for elem in contains_words1:
						result_lcs.remove(elem)
					first_sentence_list.pop(count)
				else:
					resume= resume + sentence_list[count] + ". "
					for elem in contains_words3:
						result_lcs.remove(elem)
					sentence_list.pop(count)			

				if not result_lcs:
					break;	
				
				contains_words1=[]
				contains_words3=[]
				

			# resume = ' '.join([word for word in word_tokenize(resume.translate(table))
			#   								if word not in en_stopwords])	
			# import ipdb;ipdb.set_trace()

			print("Resumen: ", resume)




			#para el primer documento		
			# for sentence in first_sent_tokenize_list: 
			# 	sentence = ' '.join([word for word in word_tokenize(sentence.translate(table))
			#   						if word not in en_stopwords]) 
			# 	sentence=sentence.split(" ")				
			# 	for elem in first_document:					
			# 		if tokenized_phrase1[elem] in sentence:
			# 			contains_words1.append(tokenized_phrase1[elem])
			# 	#	import ipdb;ipdb.set_trace()	
			# 	contains_words2.append(contains_words1)	#guarda las las palabras de la lcs que salen por oracion
			# 	score_doc1.append(len(contains_words1)/len(sentence)) # guarda los scores
			# 	contains_words1=[]						
			# #para el segundo documento						
			# for sentence in sent_tokenize_list: 
			# 	sentence = ' '.join([word for word in word_tokenize(sentence.translate(table))
			#   						if word not in en_stopwords]) 
			# 	sentence=sentence.split(" ")				
			# 	for elem in first_document:										
			# 		if tokenized_phrase2[elem] in sentence:
			# 			contains_words3.append(tokenized_phrase2[elem])
							
			# 	score_doc2.append(len(contains_words3)/len(sentence))	#guarda los scores			
			# 	contains_words4.append(contains_words3) #guarda las las palabras de la lcs que salen por oracion
			# 	contains_words3=[]	
			# import ipdb;ipdb.set_trace()
			
					

								
			
			   




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