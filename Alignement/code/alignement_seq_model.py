import sys
import random
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import fastText
import nltk
from nltk import sent_tokenize,word_tokenize
# from gensim.models import Word2Vec
import numpy as np
import json
import sys
import time
from nltk.stem import PorterStemmer
ps = PorterStemmer()

#
#Ce programme vise à faire un systeme temoins qui selectionne le dernier Np de la meilleure phrase  e la plus proche de la question.
#

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'


time1 = time.time()


with_steming_param = True
k_best_sentences = 1



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

with_steming = True
def align(question,sentence):
    span =""
    questions_words = nltk.word_tokenize(question)
    # questions_words = (list(map(ps.stem, questions_words)) if with_steming else questions_words)
    sentence_words = nltk.word_tokenize(sentence)
    # sentence_words = (list(map(ps.stem, sentence_words)) if with_steming else sentence_words)
    for sentence_word in sentence_words :
        if not(sentence_word in questions_words):
            span += sentence_word +" "
    return span
#

grammar = ('''
    NP: {<DT>?<JJ>*<NN>} # NP
    ''')
chunkParser = nltk.RegexpParser(grammar)

################################################################################################

out_json = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                if num_quest % 1000 == 0 :
                    print(num_quest)
                list_ans = []
                list_ans = get_best_sentence(model, sent_tokenize(paragraph['context']), question['question'], k_best_sentences)
                best_phrase = sent_tokenize(paragraph['context'])[list_ans[0]-1]
                list_words = word_tokenize(best_phrase)
                out_json[question['id']] = align( question['question'], paragraph['context'] )

                print(question['question'])
                print(question['answers'])
                time.sleep(5)
with open(path_dest + 'data_toTest_Alignement.json', 'w') as outfile:
    json.dump(out_json, outfile)


print("temps :", time.time() - time1)



question = "What is another  form   of precipitation besides drizzle, rain, snow, sleet and hail ? "
sentence = "The  main forms of precipitation include  drizzle, rain, sleet, snow, graupel and hail"
print(align(question,sentence))