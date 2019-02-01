'''
On va mesurer les résultats obtenu avec le système d'entitées nommées avec Scipy
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import json
import time
import spacy
import os
from score_class import *
from utils_function import *
from interesting_entities import *
from segtok.segmenter import split_single

file_name = os.path.basename(__file__)[:-3]

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_results/'
selected_data = "dev"
# selected_data = "train"

chosen_model = Function_prediction('first')
########################################################################################################
file_name = file_name + '_1_PRG.csv'
description_file_str =  "Pour chaque question on retourne la premiere entité nommée du paragraphe, on evalue le score d'exact match, f1 et le nombre d'entité inclu dans une des réponses"
########################################################################################################



time1 = time.time()
nlp = spacy.load('en_core_web_sm')

score_model = Score()


num_quest = 0
with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                question_str = question['question']
                type_question = ClassifyQuestion(question_str)
                if not type_question in list_type_question_interesting:
                    continue
                num_quest += 1
                if num_quest % 100 == 0:
                    # print(dict_type_question)
                    with open(path_dest + file_name, 'w') as f:
                        f.write(score_model.__str__())

                    print(num_quest)
                    print("temps execution", time.time() - time1)

                nlp_paragraph = nlp(paragraph['context'])
                list_predictions_data = []
                for ent in nlp_paragraph.ents:
                    if ent.label_ in interesting_entities(type_question):
                        list_predictions_data.append(normalize_answer(ent.text))

                prediction = ''
                if len(list_predictions_data) > 0:
                    prediction = list_predictions_data[0]

                score_model.add_score(type_question, prediction, list_predictions_data,  question['answers'], float(len(paragraph['context'])))



with open(path_dest + 'description.txt', 'a') as f:
    f.write(file_name+" : " + description_file_str + "\n")


with open(path_dest + file_name, 'w') as f:
    f.write(score_model.__str__())

print("temps execution", time.time()-time1)
