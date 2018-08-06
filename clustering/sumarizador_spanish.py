#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import duc
import os
import locale
import sys
import xml.etree.ElementTree as ET
import nltk.data
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from processing import *
from lcs import *
from freeling import *
from string import punctuation
es_sent_tokenize = nltk.data.load('tokenizers/punkt/spanish.pickle')
def freeling_preprocess(raw_doc):
    '''returns a list of sentences with their tokenizations'''
    client = FreelingClient('localhost', 5005)
    results = []
    all_tokens = []
    sentences = es_sent_tokenize.tokenize(raw_doc)
    for sentence in sentences:
        tokenized = client.sent_process(sentence)
        results.append(tokenized)
        all_tokens += tokenized
    return results, all_tokens
locale.setlocale(locale.LC_ALL, 'es_ES.utf8')
sys.setrecursionlimit(10000)
data_folder_original = "../spanish"



cluster_ids = os.listdir(data_folder_original)
documents = {}
contains_words1=[]
contains_words2=[]
contains_words3=[]
contains_words4=[]
score_doc1=[]
score_doc2=[]
sentence_list=[]
result_lcs=[]
resume=""
resume_printf=""
count_printf=0
#import ipdb;ipdb.set_trace()
# Gets the tokens
for cluster in cluster_ids:
	
	cluster_folder = data_folder_original + "/" + cluster
	documents[cluster] = {}
	var=0

	for document in os.listdir(cluster_folder):				
		document_path = cluster_folder + "/" + document		
		original_text = open(document_path, encoding='utf-8').read()
		original_text = unicodedata.normalize("NFKD", original_text)
		original_text = original_text.replace(u'\ufeff', u' ')
		original_text = original_text.replace(u'\n', u' ')	
		# original_text.encode('raw_unicode_escape').decode('utf-8')			
		table = str.maketrans(dict.fromkeys(punctuation))
		sent_tokenize_list = es_sent_tokenize.tokenize(original_text)			
		sentence_list, tokens = freeling_preprocess(original_text)
		#import ipdb;ipdb.set_trace()								 
		#import ipdb;ipdb.set_trace()			      
		if var==0:
			first_document=tokens
			first_sentence_list=sentence_list
			first_original_sentence_list=sent_tokenize_list
			sentence_list=[]
			var=1 
		else:
			#import ipdb;ipdb.set_trace() 
			if var==2:
				first_document=resume_printf
				first_original_sentence_list = es_sent_tokenize.tokenize(resume_printf)					
				first_sentence_list, tokens_resume = freeling_preprocess(first_document)	
				#import ipdb;ipdb.set_trace()	
				tokenized_phrase1 = resume_printf.split(" ")
				tokenized_phrase2 = tokens
				resume=""
				resume_printf=""

			else:
				var=2	
				tokenized_phrase1 = first_document
				tokenized_phrase2 = tokens
				
			# tokenized_phrase1 = "Hay una abeja Hay una flor La abeja hace miel La miel es cara".split(" ")
			# tokenized_phrase2 = "La abeja va hasta flor para hacer miel que se vende cara en el mercado".split(" ")
			# first_sentence_list = ["Hay una abeja","Hay una flor","La abeja hace miel","La miel es cara"]
			# sentence_list = ["La abeja va hasta flor para hacer miel que se vende cara en el mercado"]			
			result_lcs=lcs(tokenized_phrase1, tokenized_phrase2, " ").split(" ")
			count=0				
			while count == 0:
				if not result_lcs:
					break;
				else:	
					if first_sentence_list:
						contains_words1=lcs(result_lcs,first_sentence_list[count]," ").split(" ")
						if '' in contains_words1:
							contains_words1.remove('')
						score_doc1=len(contains_words1)/len(first_sentence_list[count])	
					if sentence_list:
						contains_words3=lcs(result_lcs,sentence_list[count], " ").split(" ")	
						if '' in contains_words3:
							contains_words3.remove('')		
						score_doc2=len(contains_words3)/len(sentence_list[count])
					if score_doc1 > score_doc2:											
						for elem in contains_words1:
							if elem in result_lcs:	
								result_lcs.remove(elem)
								flag=1
						if flag==1:
							#resume= resume + first_sentence_list[count]
							resume_printf = resume_printf + first_original_sentence_list[count]
							flag=0
															
						first_original_sentence_list.pop(count)		
						first_sentence_list.pop(count)
					else:												
						for elem in contains_words3:
							if elem in result_lcs:
								result_lcs.remove(elem)
								flag=1
						if flag==1:
							#resume= resume + sentence_list[count]
							resume_printf = resume_printf + sent_tokenize_list[count]
							flag=0
						
						sentence_list.pop(count)
						sent_tokenize_list.pop(count)			

						
					#import ipdb;ipdb.set_trace()
					contains_words1=[]
					contains_words3=[]
					score_doc1=0
					score_doc2=0

			count_printf+=1	
			if count_printf==9:
			# 	import ipdb;ipdb.set_trace()
				print("Resumen: ", resume_printf)
			#	count_printf=0	
			first_sentence_list=[]
			sentence_list=[]
			



