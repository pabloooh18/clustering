import duc
import json


data_folder_original = "../res/original"
data_folder_sum10 = "../res/sum10"
data_folder_sum100 = "../res/sum100"

output_file = "../res/output.txt"

# Rouge centroids
documents = duc.get_rouge_document_clusters(data_folder_original)
rouge_space = duc.get_vector_space_from_clusters(documents)
vectorized_documents = duc.convert_to_vectors(documents, rouge_space)
centroids = duc.get_cluster_centroids(vectorized_documents)
import ipdb;ipdb.set_trace()
