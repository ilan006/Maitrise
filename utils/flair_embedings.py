from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# initialize the document embeddings, mode = mean
pool_embeddings = DocumentPoolEmbeddings([glove_embedding, flair_embedding_backward, flair_embedding_forward])

# create an example sentence
sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
pool_embeddings.embed(sentence)

def get_fastText_embeding(sentence):
    pass
    # return