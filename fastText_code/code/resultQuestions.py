import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import fastText
from nltk import sent_tokenize
# from gensim.models import Word2Vec
import numpy as np
import json
import sys
import time
from nltk.stem import PorterStemmer
ps = PorterStemmer()

#
#Ce programme vise à evaluer le nombre de phrase bien retourner en fonction du type de question
#

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_perso/'

time1 = time.time()


with_steming_param = False
k_best_sentences = 1



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



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
                out_json[question['id']] = [list_ans, ClassifyQuestion(question['question'])]

with open(path_dest + 'data_toTest_fasText.json', 'w') as outfile:
    json.dump(out_json, outfile)


with open(path_dest + 'data_Dev.json', 'r') as input:
    data_ref = json.load(input)
    input.close()

with open(path_dest + 'data_toTest_fasText.json', 'r') as input:
    data_to_evaluate = json.load(input)
    input.close()

sum = 0
dict_classificateur_question = {}
dict_total_question = {}
for id in data_ref:
    j = 1
    for position in data_ref[id]:
        j += 1
        try:
            dict_total_question[data_to_evaluate[id][1]] += 1
        except:
            dict_total_question[data_to_evaluate[id][1]] = 1
        if position in data_to_evaluate[id][0]:
            sum += 1
            try:
                dict_classificateur_question[data_to_evaluate[id][1]] += 1
            except:
                dict_classificateur_question[data_to_evaluate[id][1]] = 1
            break



print("classification")
print(dict_classificateur_question)
print("total")
print(dict_total_question)

print(sum / float(len(data_ref)))

print("temps :", time.time() - time1)


