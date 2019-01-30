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
file_name = os.path.basename(__file__)[:-3]

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_results/'
selected_data = "dev"
# selected_data = "train"


########################################################################################################
file_name = file_name + '_1.csv'
description_file_str =  "Pour chaque question on retourne la premiere entité nommée du paragraphe, on evalue le score d'exact match, f1 et le nombre d'entité inclu dans une des réponses"
########################################################################################################

def interesting_entities(type_question):
    if type_question == 'Where?':
        interisting_entities = ("GPE", "LOC", "FAC", "ORG")
    elif type_question == 'How much / many?' :
        interisting_entities = ("MONEY","QUANTITY","PERCENT", "CARDINAL", "TIME","DATE", "ORDINAL")
    elif type_question == 'What name / is called?':
        interisting_entities = ("PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE",'FAC')
    elif type_question == 'Who?':
        interisting_entities = ("PERSON","ORG","NORP","GPE","PRODUCT")
    elif type_question == 'When / What year?':
        interisting_entities = ("TIME","DATE","EVENT")
    return interisting_entities


time1 = time.time()
nlp = spacy.load('en_core_web_sm')

score_model = Score()

list_type_question_interesting = ['Where?', 'How much / many?', 'What name / is called?', 'Who?', 'When / What year?']
dict_type_question = {}
for type_question in list_type_question_interesting:
    dict_type_question[type_question] = [0, 0, 0, 0, 0, 0, 0]
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
                    print(num_quest)


                    with open(path_dest + file_name, 'w') as f:
                        f.write(score_model.__str__())
                    print("save")
                    print("temps execution", time.time() - time1)



                nlp_paragraph = nlp(paragraph['context'])
                list_predictions_data = []
                for ent in nlp_paragraph.ents:
                    if ent.label_ in interesting_entities(type_question):
                        list_predictions_data.append(normalize_answer(ent.text))

                prediction = list_predictions_data[0]
                score_model.add_score(type_question, prediction, list_predictions_data,  question['answers'], float(len(paragraph['context'])))



with open(path_dest + 'description.txt', 'a') as f:
    f.write(file_name+" : " + description_file_str + "\n")



with open(path_dest + file_name, 'w') as f:
    f.write(score_model.__str__())

print("temps execution", time.time()-time1)
