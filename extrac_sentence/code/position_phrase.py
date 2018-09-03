# coding= utf-8

import os, sys
""" Official evaluation script for v1.1 of the SQuAD dataset. """
from collections import Counter
import string
import re
import argparse
import json
import sys
from nltk import sent_tokenize
import operator
import pickle
path_data = '../../data/'

sys.path.append("../../")
from utils.get_numeral_sentence import numeral_sentence


dict = {}
nb_paragraphe = 0
taille_max_parg = 0.0
total_answer = 0
with open(path_data + 'train-v1.1.json') as json_data:
    d = json.load(json_data)
    json_data.close()
    # print('title : ' + d['data'][0]['title'])
    for data in d['data']:
        for paragraph in data['paragraphs']:
            nb_paragraphe += 1
            taille_parag = len(sent_tokenize(paragraph['context']))
            if taille_parag > taille_max_parg :
                taille_max_parg = taille_parag
                # print(taille_parag)
            for question in paragraph['qas']:
                list_ans =[]
                for answer in question['answers']:
                    if total_answer % 1000 == 0: print(total_answer)
                    pos = answer['answer_start']
                    if pos not in list_ans :
                        list_ans.append(pos)
                        total_answer += 1
                        sentence_positionRelative = round(numeral_sentence(pos, paragraph['context'],bool_normalize = True), 1)
                        if sentence_positionRelative in dict:
                            dict[sentence_positionRelative] += 1
                        else:
                            dict[sentence_positionRelative] = 1
    sum = 0
    for key in dict:
        print(str(key) +"  :  " + str(dict[key] / float(87599)))
        sum += dict[key]
    print(' nombre de paragraphe au total : ' + str(sum))
    print("nombre de phrase moyen par paragraphe : " + str(float(sum) / nb_paragraphe))

# pickle.dump(dict, open("../dict/freq_sentence_positionRelative.pickle", "wb"))
    # print " "
    # print(d['data'][0]['paragraphs'][0]['qas'][0]['question'])
    # print d['data'][0]['paragraphs'][0]['qas'][1]['answers'][0]['text']
    # print d['data'][0]['paragraphs'][0]['qas'][1]['answers'][0]['answer_start']
    # for key in d['data'][0]['paragraphs'][0]['qas'][0]['question']:
    #     print key


    # for key in d['data'][0]:
    #     print key
    # print len(d['data'][0])
    #print(d['data'][0]['paragraphs'])

    # print(len(d['data']))
    # for i in range(len(d['data'])):
    #     print(d['data'][i]['title'])

list_dict_trie =sorted(dict.items(), reverse=False, key=operator.itemgetter(0))

with open('../data_perso/freq_position_train.csv', 'w') as f:
    item_str = "position relative"+","+"comptage relatif"+","+"comptage"+","+"class" + "\n"
    f.write(item_str)
    for item in list_dict_trie:
        item_str = str(round(max(item[0]-0.05,0.0),2))+"-"+str(round(min(item[0]+0.05,1.0),2))+","+ str(item[1] / float(total_answer))+","+ str(item[1] )+",1" + "\n"
        f.write(item_str)
