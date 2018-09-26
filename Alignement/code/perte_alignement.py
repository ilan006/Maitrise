'''
ce systeme va mesuré la perte réaliser par le passage dans l'aligneur
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
from nltk.stem import PorterStemmer
ps = PorterStemmer()

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'


time1 = time.time()

total_dedans_possible = 0
total_dedans_sorti = 0
out_json = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()
with open(path_dest + 'data_toTest_Alignement.json', 'r') as outfile:
    data_test = json.load(outfile)
    outfile.close()
    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                if num_quest % 1000 == 0:
                    print(num_quest)
                list_ans = get_best_sentence(model, sent_tokenize(paragraph['context']), question['question'], 1)
                best_phrase = sent_tokenize(paragraph['context'])[list_ans[0] - 1]
                for answer in question['answers']:
                    if (answer['text'] in best_phrase):
                        total_dedans_possible += 1
                        if (answer['text'] in data_test[question['id']]):
                            total_dedans_sorti += 1


                # print("question :   ", question['question'])
                # print("meilleure phrase :   ",best_phrase)
                # print("span retourné :   ", out_json[question['id']])
                # print("reponses atendues :   ", question['answers'])
                # print("    ")
                # time.sleep(5)


print("total_dedans_possible :", total_dedans_possible)
print("total_dedans_sorti :", total_dedans_sorti)
print("ratio :", total_dedans_sorti/float(total_dedans_possible))
print("temps :", time.time() - time1)
