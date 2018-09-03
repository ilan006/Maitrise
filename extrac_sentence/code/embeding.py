import numpy as np
import os
from random import shuffle
import re
from nltk import sent_tokenize
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
import string
path_data = '../../data/'
path_dest = '../../data_perso/'
with open(path_dest + 'train_concat.txt', 'r') as f:
    input_text = f.read()

# remove parenthesis
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
# store as list of sentences
tab_sent = sent_tokenize(input_text_noparens)

ps = PorterStemmer()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


sentences_segm = []
for sent_str in tab_sent:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    # tokens = sent_str.lower().split()
    tokens = list(map(ps.stem,(normalize_answer(sent_str).split())))
    sentences_segm.append(tokens)


model = Word2Vec(sentences=sentences_segm, size=100, window=5, min_count=5, workers=16, sg=0)

model.save(path_dest + 'embeding.txt')
# sentence_1_avg_vector = avg_feature_vector(sentence_1.split(), model=word2vec_model, num_features=100)
# a = model.vector_size
# print(a)
# print(model.wv['queen'])