from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

# # initialize the word embeddings
# glove_embedding = WordEmbeddings('glove')
# flair_embedding_forward = FlairEmbeddings('news-forward')
# flair_embedding_backward = FlairEmbeddings('news-backward')
fastText_embedding = WordEmbeddings('en-wiki')
# initialize the document embeddings, mode = mean
# pool_embeddings = DocumentPoolEmbeddings([glove_embedding, flair_embedding_backward, flair_embedding_forward])

pool_embeddings = DocumentPoolEmbeddings([fastText_embedding])
# pool_embeddings = DocumentPoolEmbeddings([flair_embedding_forward,flair_embedding_backward])
# def get_fastText_embeding(text_str):
#     '''
#     Fonction qui retourne l'embedding d'un
#     :param text_str: text a embeddé
#     :return: le vecteur d'embeding correspondant au text
#     '''
#     text = Sentence(text_str)
#     pool_embeddings.embed(text)
#     return text.get_embedding()

