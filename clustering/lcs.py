import copy
import duc
import os
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from processing import *
from string import punctuation


data_folder_original = "../res/original"

def lcs_matrix(s1, s2):
    matrix=[[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i == 0 or j == 0:
                    matrix[i][j] = 0
            elif s1[i-1] == s2[j-1]:
                matrix[i][j] = matrix[i-1][j-1] + 1
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])
    return matrix

def backtrack_lcs_matrix(matrix, s1, s2, i, j):
    if i == 0 or j == 0:
        return []

    if s1[i-1] == s2[j-1]:
        return backtrack_lcs_matrix(matrix, s1, s2, i-1, j-1) + [s1[i-1]]
    if matrix[i][j-1] > matrix[i-1][j]:
        return backtrack_lcs_matrix(matrix, s1, s2, i, j-1)
    return backtrack_lcs_matrix(matrix, s1, s2, i-1, j)

def all_backtrack_lcs_matrix(matrix, s1, s2, i, j):
    if i == 0 or j == 0:
        return [[]]
    elif s1[i-1] == s2[j-1]:
        return [s + [s1[i-1]] for s in all_backtrack_lcs_matrix(matrix,
                                                                s1, s2,
                                                                i-1, j-1)]
    else:
        result = []
        if matrix[i][j-1] >= matrix[i-1][j]:
            result = result + all_backtrack_lcs_matrix(matrix,
                                                       s1, s2,
                                                       i, j-1)
        if matrix[i-1][j] >= matrix[i][j-1]:
            result = result + all_backtrack_lcs_matrix(matrix,
                                                       s1, s2,
                                                       i-1, j)
        return result

def lcs(s1, s2, joiner=""):
    matrix = lcs_matrix(s1, s2)
    return joiner.join(backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)))

def all_lcs(s1, s2, joiner=""):
    matrix = lcs_matrix(s1, s2)
    result = []
    for elem in all_backtrack_lcs_matrix(matrix, s1, s2, len(s1), len(s2)):
        result.append(joiner.join(elem))
    return list(set(result))


cluster_ids = os.listdir(data_folder_original)
documents = {}
var=0
#import ipdb;ipdb.set_trace()
# Gets the tokens
for cluster in cluster_ids:
    cluster_folder = data_folder_original + "/" + cluster
    documents[cluster] = {}
    for document in os.listdir(cluster_folder):
        document_path = cluster_folder + "/" + document
        tree = ET.parse(document_path)
        root = tree.getroot()
        original_text = root.find("TEXT").text
        lemmatized = lemmatize(original_text)
        table = str.maketrans(dict.fromkeys(punctuation))
        en_stopwords = set(stopwords.words('english'))
        tokens = lemmatized.translate(table)    
        if var==1:
            #import ipdb;ipdb.set_trace() #si utilizo el debug funciona sino me tira error¿?
            tokenized_phrase1 = first_document.split(" ")
            tokenized_phrase2 = tokens.split(" ")
            print("LCS:", lcs(tokenized_phrase1, tokenized_phrase2, " "))
            #print("All LCS:", all_lcs(first_document, tokens, " "))
            var=0
        if var==0:
            first_document=tokens
            var=1    
                                                       
      

#import ipdb;ipdb.set_trace()
#funcion de prueba    
#documents = duc.get_rouge_document(data_folder_original)

# tokenized_phrase1 = "la fea casa es verde y fria".split(" ")
# tokenized_phrase2 = "la casa fea es fria y verde".split(" ")

# str1 = "abcabcaa"
# str2 = "acbacba"

# print("De '" + str1 + "' y '" + str2 + "':")
# print("LCS:", lcs(str1, str2))
# print("All LCS:", all_lcs(str1, str2))
# print("De '" + " ".join(tokenized_phrase1) + "' y '" +\
#       " ".join(tokenized_phrase2) + "':")
# print("LCS:", lcs(tokenized_phrase1, tokenized_phrase2, " "))
# print("All LCS", all_lcs(tokenized_phrase1, tokenized_phrase2, " "))
