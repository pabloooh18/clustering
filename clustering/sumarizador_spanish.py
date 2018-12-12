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
from sklearn.feature_extraction.text import TfidfVectorizer
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
# data_folder_original = "../spanish"
data_folder_original = "../esoriginal"


def summary_wordlimit(summary_sentences, sentence_scores, word_limit):
	limited_summary = []
	current_length = 0
	score_position = [(i,score) for i, score in enumerate(sentence_scores)]
	score_position.sort(key=lambda x:x[1], reverse=True)
	#import ipdb;ipdb.set_trace()
	for position, score in score_position:
		raw_sent= summary_sentences[position]
		if (current_length + raw_sent.count(" ") + 1) <= word_limit:
			limited_summary.append(position)
			current_length += raw_sent.count(" ") + 1
	limited_summary.sort()
	limited_summary = [summary_sentences[pos] for pos in limited_summary]
	#import ipdb;ipdb.set_trace()
	return limited_summary

def summary_limit(summary_sentences, sentence_scores, byte_limit):
	limited_summary = []
	current_length = 0
	score_position = [(i,score) for i, score in enumerate(sentence_scores)]
	score_position.sort(key=lambda x:x[1], reverse=True)

	for position, score in score_position:
		raw_sent= summary_sentences[position]
		if (current_length + len(raw_sent.encode("utf-8"))) <= byte_limit:
			limited_summary.append(position)
			current_length += len(raw_sent.encode("utf-8"))
	limited_summary.sort()
	limited_summary = [summary_sentences[pos] for pos in limited_summary]
	return limited_summary 


cluster_ids = os.listdir(data_folder_original)
documents = {}
contains_words1=[]
contains_words2=[]
contains_words3=[]
contains_words4=[]
score_doc1=0
score_doc2=0
sentence_list=[]
result_lcs=[]
summary_sentence_scores=[]
resume=""
resume_printf=[]
new_resume=""
count_printf=0
#import ipdb;ipdb.set_trace()
# Gets the tokens
for cluster in cluster_ids:
	
	cluster_folder = data_folder_original + "/" + cluster
	var=0

	document_path = cluster_folder + "/" + os.listdir(cluster_folder)[0]
	print(document_path)
	tree = ET.parse(document_path)
	root = tree.getroot()
	for i, news_item in enumerate(root):
		#print(i+1, ": ", news_item.find("title").text)
		original_text = news_item.find("content").text
	# cluster_folder = data_folder_original + "/" + cluster
	# documents[cluster] = {}	
	# for document in os.listdir(cluster_folder):				
	# 	document_path = cluster_folder + "/" + document		
	# 	original_text = open(document_path, encoding='utf-8').read()
	# 	original_text = unicodedata.normalize("NFKD", original_text)
	# 	original_text = original_text.replace(u'\ufeff', u' ')
	# 	original_text = original_text.replace(u'\n', u' ')	
	# 	# original_text.encode('raw_unicode_escape').decode('utf-8')			
	# 	table = str.maketrans(dict.fromkeys(punctuation))
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
				# import ipdb;ipdb.set_trace()
				first_document=" ".join(resume_printf)
				first_original_sentence_list = es_sent_tokenize.tokenize(" ".join(resume_printf))					
				first_sentence_list, tokens_resume = freeling_preprocess(first_document)	
				#import ipdb;ipdb.set_trace()	
				tokenized_phrase1 = first_document.split(" ")
				tokenized_phrase2 = tokens
				resume=""
				resume_printf=[]
				summary_sentence_scores=[]

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
						#import ipdb;ipdb.set_trace()
						# if contains_words1:	
						# 	tfidf=TfidfVectorizer(vocabulary=sorted(set(contains_words1)),ngram_range=(1,2))
						# 	tfs=tfidf.fit_transform(first_sentence_list[count])
						# 	rows, cols = tfs.nonzero()
						# 	for row, col in zip(rows, cols):
						# 		score_doc1=score_doc1+tfs[row,col]
						# else:
						# 	score_doc1=0	
						score_doc1=len(contains_words1)/len(first_sentence_list[count])	
					if sentence_list:
						contains_words3=lcs(result_lcs,sentence_list[count], " ").split(" ")	
						if '' in contains_words3:
							contains_words3.remove('')
						# if contains_words3:	
						# 	tfidf=TfidfVectorizer(vocabulary=sorted(set(contains_words3)),ngram_range=(1,2))
						# 	tfs=tfidf.fit_transform(sentence_list[count])
						# 	rows, cols = tfs.nonzero()
						# 	for row, col in zip(rows, cols):
						# 		score_doc2=score_doc2+tfs[row,col]
						# else:
						# 	score_doc2=0			
						score_doc2=len(contains_words3)/len(sentence_list[count])
					if score_doc1 > score_doc2:											
						for elem in contains_words1:
							if elem in result_lcs:	
								result_lcs.remove(elem)
								flag=1
						if flag==1:
							#resume= resume + first_sentence_list[count]
							resume_printf.append(first_original_sentence_list[count])
							summary_sentence_scores.append(score_doc1)
							flag=0
															
						first_original_sentence_list.pop(count)		
						first_sentence_list.pop(count)
					else:
						if score_doc1 == score_doc2:	
							if sentence_list:
								for elem in contains_words3:
									if elem in result_lcs:
										result_lcs.remove(elem)
										flag=1
								if flag==1:
									# resume= resume + sentence_list[count]
									resume_printf.append(sent_tokenize_list[count])
									summary_sentence_scores.append(score_doc2)
									flag=0

								sentence_list.pop(count)
								sent_tokenize_list.pop(count)
								
							else:	
								if first_sentence_list:
									for elem in contains_words1:
										if elem in result_lcs:	
											result_lcs.remove(elem)
											flag=1
									if flag==1:
										# resume= resume + first_sentence_list[count]
										resume_printf.append(first_original_sentence_list[count])
										summary_sentence_scores.append(score_doc1)
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
								resume_printf.append(sent_tokenize_list[count])
								summary_sentence_scores.append(score_doc2)	
								flag=0

							sentence_list.pop(count)
							sent_tokenize_list.pop(count)
									

						
					#import ipdb;ipdb.set_trace()
					contains_words1=[]
					contains_words3=[]					
					score_doc1=0
					score_doc2=0
															
						

			# count_printf+=1	
			# if count_printf==9:
			# import ipdb;ipdb.set_trace()
			for_printf=""
			for_printf=" ".join(summary_wordlimit(resume_printf,summary_sentence_scores,100))

			# para imprimir en carpeta

			# folder_path = "../summaries/%s" % (cluster)
			# try:
			# 	os.makedirs(folder_path)
			# except:
			# 	pass
			# with open("%s/%s_%s.txt" % (folder_path, i, count_printf), "w") as f: f.write(for_printf)
			print("\n\nResumen: ", for_printf)
				# count_printf=0
			# count_printf+=1	
			# if count_printf==9:
			# 	#import ipdb;ipdb.set_trace()
			# 	contains_words1=[]
			# 	contains_words2=[]
			# 	contains_words3=[]
			# 	contains_words4=[]
			# 	score_doc1=0
			# 	score_doc2=0
			# 	sentence_list=[]
			# 	result_lcs=[]
			# 	summary_sentence_scores=[]
			# 	resume=""
			# 	resume_printf=[]
			# 	new_resume=""
			# 	count_printf=0
			# 	var=0		
			first_sentence_list=[]
			sentence_list=[]
			



