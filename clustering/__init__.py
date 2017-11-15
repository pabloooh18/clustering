import duc

data_folder_original = "../res/original"
data_folder_sum10 = "../res/sum10"
data_folder_sum100 = "../res/sum100"

# Text dumps
out_documents_file = "../res/rouge_docs.txt"
out_rouge_space_file = "../res/rouge_space.txt"
out_vectorized_documents_file = "../res/rouge_vect_docs.txt"
out_centroids_file = "../res/rouge_centroids.txt"

#Pickled dumps
p_documents_file = "../res/rouge_docs.p"
p_rouge_space_file = "../res/rouge_space.p"
p_vectorized_documents_file = "../res/rouge_vect_docs.p"
p_centroids_file = "../res/rouge_centroids.p"

# Rouge centroids
documents = duc.get_rouge_document_clusters(data_folder_original)
duc.dump(documents, out_documents_file)
duc.pickle_dump(documents, p_documents_file)

rouge_space = duc.get_vector_space_from_clusters(documents)
duc.dump(rouge_space, out_rouge_space_file)
duc.pickle_dump(rouge_space, p_rouge_space_file)

vectorized_documents = duc.convert_to_vectors(documents, rouge_space)
duc.dump(vectorized_documents, out_vectorized_documents_file)
duc.pickle_dump(vectorized_documents, p_vectorized_documents_file)

centroids = duc.get_cluster_centroids(vectorized_documents)
duc.dump(centroids, out_centroids_file)
duc.pickle_dump(centroids, p_centroids_file)
