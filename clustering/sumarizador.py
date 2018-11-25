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

sys.setrecursionlimit(5000)
data_folder_original = "../res/original"

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
summary_sentence_scores=[]
resume=""
resume_printf=[]
new_resume=""
count_printf=0

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
			first_original_sentence_list=sent_tokenize_list
			sentence_list=[]
			var=1 
		else:
			#import ipdb;ipdb.set_trace() 
			if var==2:
				first_document=resume
				#import ipdb;ipdb.set_trace()
				
				new_resume=" ".join(resume_printf)

				#import ipdb;ipdb.set_trace()	
				first_original_sentence_list = sent_tokenize(new_resume)
				for sentence in first_original_sentence_list: #quita puntuacion y stopwords de cada oracion			
					sentence = ' '.join([word for word in word_tokenize(sentence.translate(table))
				  						if word not in en_stopwords])
					first_sentence_list.append(sentence)
				#import ipdb;ipdb.set_trace()	
				tokenized_phrase1 = first_document.split(" ")
				tokenized_phrase2 = tokens.split(" ")
				resume=""
				resume_printf=[]
				summary_sentence_scores=[]

			else:
				var=2	
				tokenized_phrase1 = first_document.split(" ")
				tokenized_phrase2 = tokens.split(" ")
				
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
						contains_words1=lcs(result_lcs,first_sentence_list[count].split(" ")," ").split(" ")
						if '' in contains_words1:
							contains_words1.remove('')
						score_doc1=len(contains_words1)/len(first_sentence_list[count].split(" "))	
					if sentence_list:
						contains_words3=lcs(result_lcs,sentence_list[count].split(" "), " ").split(" ")	
						if '' in contains_words3:
							contains_words3.remove('')		
						score_doc2=len(contains_words3)/len(sentence_list[count].split(" "))
					if score_doc1 > score_doc2:											
						for elem in contains_words1:
							if elem in result_lcs:	
								result_lcs.remove(elem)
								flag=1
						if flag==1:
							resume= resume + first_sentence_list[count]
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
									resume= resume + sentence_list[count]
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
										resume= resume + first_sentence_list[count]
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
								resume= resume + sentence_list[count]
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

			count_printf+=1	
			if count_printf==9:
				for_printf=""
				for_printf=" ".join(summary_wordlimit(resume_printf,summary_sentence_scores,15))
				print("\n\n\nResumen de "+ cluster +": " +for_printf)
				count_printf=0	
			first_sentence_list=[]
			sentence_list=[]







