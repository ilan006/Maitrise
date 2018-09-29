'''
Ce programme vise à faire un systeme temoins qui selectionne le dernier Np des n meilleurs phrase
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import fastText
import nltk
from nltk import sent_tokenize,word_tokenize
import json
import time


model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'


time1 = time.time()


with_steming_param = False
k_best_sentences = 5


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
                list_phrases = [sent_tokenize(paragraph['context'])[i-1] for i in list_ans]
                span=[]
                best_cosine_sim = 0
                for phrase in list_phrases:
                    list_words = word_tokenize(phrase)
                    tagged = nltk.pos_tag(nltk.word_tokenize(phrase))
                    tree = chunkParser.parse(tagged)
                    for subtree in tree.subtrees():
                        if (subtree.label() == "NP"):
                            span_test = [leave[0] for leave in subtree.leaves()]
                    cosine_test = cosine_similarity(avg_sentence_vector(" ".join(span_test), model), avg_sentence_vector(question['question'], model))
                    if best_cosine_sim < cosine_test:
                        best_cosine_sim = cosine_test
                        span = span_test
                out_json[question['id']] = ' '.join(span)


with open(path_dest + 'data_toTest_System3.json', 'w') as outfile:
    json.dump(out_json, outfile)


print("temps :", time.time() - time1)


