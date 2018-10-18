'''
Systeme temoin qui selectionne aleatoirement un span de mot dans la phrase la plus proche de la question.
'''

import sys
import random
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import fastText
from nltk import sent_tokenize,word_tokenize
import json
import time


model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'


time1 = time.time()


k_best_sentences = 1
taille_span = 2

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
                best_phrase = sent_tokenize(paragraph['context'])[list_ans[0]]
                list_words = word_tokenize(best_phrase)
                num_words = random.randrange(0, len(list_words)-taille_span+1)
                span = list_words[num_words:num_words+taille_span]
                out_json[question['id']] = ' '.join(span)

with open(path_dest + 'data_toTest_System1.json', 'w') as outfile:
    json.dump(out_json, outfile)


print("temps :", time.time() - time1)


