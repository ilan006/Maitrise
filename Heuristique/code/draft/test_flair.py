# from flair.models import SequenceTagger
# from flair.data import Sentence
#
# from segtok.segmenter import split_single
#
# # your text of many sentences
# text = "This is a sentence. This is another sentence. I love Berlin. I love paris."
#
# # use a library to split into sentences
# sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(text)]
#
#
# # predict tags for list of sentences
# tagger: SequenceTagger = SequenceTagger.load('ner')
# tagger.predict(sentences)
#
# for sentence in sentences:
#     for entity in sentence.get_spans('ner'):
#         print(entity)
#         print(entity.tag)
#
#
# sentence = Sentence('I love Berlin.')
# # predict NER tags
# tagger.predict(sentence)
#
# # # print sentence with predicted tags
# # print(sentence.to_tagged_string())
#
# for entity in sentence.get_spans('ner'):
#     print(entity)
#
# from flair.models import SequenceTagger
#
# tagger = SequenceTagger.load('ner-ontonotes-fast')
# print(tagger.tag_dictionary.__dict__.keys())
# print(tagger.tag_dictionary.__dict__['idx2item'])
#
#
# tagger.predict(sentences)
#
# for sentence in sentences:
#     for entity in sentence.get_spans('ner'):
#         print(entity.text)
#         print(entity.tag)
import torch

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                              flair_embedding_backward,
                                              flair_embedding_forward])

# create an example sentence
sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
document_embeddings.embed(sentence)

# now check out the embedded sentence.
print(sentence.get_embedding())

total = 0
somme_tensors = torch.zeros(4196)
for token in sentence:
    total += 1
    somme_tensors += token.get_embedding()

print(total)
print(somme_tensors/total)