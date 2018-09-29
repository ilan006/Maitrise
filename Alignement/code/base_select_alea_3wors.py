import fastText
from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import json
import sys
import time
from nltk import ngrams
from nltk.stem import PorterStemmer
ps = PorterStemmer()

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'

time0 = time.time()
time1 = time.time()

with_steming_param = False
k_best_sentences = 1
similarity_type = 1 #1 : cosine, 2:dice

dict_question={}
class_final = "repli_phrase_n_cos"

with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

nb_question = 0
out_json = {}
dict_question = {}
for data in d['data']:
    for paragraph in data['paragraphs']:
        for question in paragraph['qas']:
            nb_question += 1
            dict_question[question['id']] = [0,0]
            if nb_question % 1000 == 0:
                print(nb_question)
                print("temps :", time.time() - time1)
                time1 = time.time()
            out_json[question['id']] = question['answers'][0]['text']

with open(path_dest + 'data_to_test.json', 'w') as outfile:
    json.dump(out_json, outfile)
    nb_question = 0



