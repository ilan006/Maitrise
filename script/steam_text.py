 
import json
import sys
from nltk import sent_tokenize
from nltk.stem import PorterStemmer
import pickle

path_data = '../data/'
path_dest = '../data_perso/'
ps = PorterStemmer()

with open(path_dest + 'train_concat.txt') as f:
    text = f.read()
    sentences = sent_tokenize(text)
    with open(path_dest + 'train_concat_steam.txt', 'w') as output_file:
        for sentence in sentences:
            list_word = list(map(ps.stem, sentence.split()))
            output_file.write(" ".join(list_word))
