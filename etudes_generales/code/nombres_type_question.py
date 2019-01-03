"""
Ce programme réalise une statistique des types de questions affin de comptabilisé les fréquence exacte que chaque type.
"""
import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import json
import sys
import time


path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'
selected_data = "train"
time1 = time.time()

out_json = {}
compte_question = {}

with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 0
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                print(question['question'])
                if num_quest % 1000 == 0:
                    print(num_quest)
                    print(time.time() - time1)
                    # time.sleep(5)
                try:
                    compte_question[ClassifyQuestion(question['question'])] += 1
                except:
                    compte_question[ClassifyQuestion(question['question'])] = 1


print("classification")
print(compte_question)


print("temps :", time.time() - time1)

with open('../resultats/nombres_type_question_'+selected_data+'.csv', 'w') as f:
    item_str = "type de question"+","+"compte relatif"+","+"compte"+","+"\n"
    f.write(item_str)
    for item in compte_question:
        item_str = str(
            str(item) + "," + str(compte_question[item]) + "," + str(compte_question[item] / float(num_quest))) + "\n"
        f.write(item_str)
