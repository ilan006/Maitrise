"""
Ce programme réalise une statistique des types de questions affin de comptabilisé les fréquence exacte que chaque type.
"""
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

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'

out_json = {}
compte_question = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                if num_quest % 1000 == 0:
                    print(num_quest)
                try:
                    compte_question[ClassifyQuestion(question['question'])] += 1
                except:
                    compte_question[ClassifyQuestion(question['question'])] = 1


print("classification")
print(compte_question)


print("temps :", time.time() - time1)


