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


file_name = os.path.basename(__file__)[:-3]

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_results/'
selected_data = "dev"
# selected_data = "train"

# chosen_model = Function_prediction('first', 'PRG_spacy')
# chosen_model = Function_prediction('first', 'SENT_spacy')
# chosen_model = Function_prediction('first', 'PRG_flair')
chosen_model = Function_prediction('first', 'SENT_flair')
########################################################################################################
file_name = file_name + '_' + chosen_model.get_type_method() + '_' + chosen_model.model + '.csv'
description_file_str = chosen_model.get_description() +' '+ chosen_model.model_description
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
                    with open(path_dest + file_name, 'w') as f:
                        f.write(score_model.__str__())

                    print(num_quest)
                    print("temps execution", time.time() - time1)

                list_predictions_data = chosen_model.get_list_predictions(paragraph['context'], type_question)

                prediction = chosen_model.predict(list_predictions_data)

                score_model.add_score(type_question, prediction, list_predictions_data,  question['answers'], float(len(paragraph['context'])))



with open(path_dest + 'description.txt', 'a') as f:
    f.write(file_name+" : " + description_file_str + "\n \n")


with open(path_dest + file_name, 'w') as f:
    f.write(score_model.__str__())

print("temps execution", time.time()-time1)
