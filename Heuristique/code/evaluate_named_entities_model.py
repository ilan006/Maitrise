'''
On va mesurer les résultats obtenu avec le système d'entitées nommées
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import json
import sys
import time
import spacy


import os
file_name = os.path.basename(__file__)[:-3]

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_results/'
selected_data = "dev"
# selected_data = "train"


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


list_type_question_interesting = ['Where?', 'How much / many?', 'What name / is called?', 'Who?', 'When / What year?']
dict_type_question= {}
for type_question in list_type_question_interesting :
    dict_type_question[type_question] = [0, 0, 0] # nb de question observees, exact_match , f1
num_quest = 0
with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                type_question = ClassifyQuestion(question['question'])
                if not type_question in list_type_question_interesting:
                    continue
                num_quest += 1
                if num_quest % 100 == 0:
                    # print(dict_type_question)
                    print(num_quest)



##################################
                    exact_match = f1 = moy_exact_match_all_types = moy_f1_all_types = total_nb_questions = 0

                    with open(path_dest + file_name + '.csv', 'w') as f:
                        f.write("Question ; nb_de_questions ; exact_match ; f1" + "\n")
                        for key in dict_type_question:
                            exact_match = 100.0 * (dict_type_question[key][1] / dict_type_question[key][0])
                            f1 = 100.0 * dict_type_question[key][2] / dict_type_question[key][0]
                            moy_exact_match_all_types += exact_match * dict_type_question[key][0]
                            moy_f1_all_types += f1 * dict_type_question[key][0]
                            total_nb_questions += dict_type_question[key][0]

                            item_str = (key, str(dict_type_question[key][0]), str(exact_match), str(f1))
                            item_str = " ; ".join(item_str) + "\n"
                            f.write(item_str)
                        item_str = ("MOYENNE TOUT TYPE", str(total_nb_questions),
                                    str(moy_exact_match_all_types / total_nb_questions),
                                    str(moy_f1_all_types / total_nb_questions))
                        item_str = " ; ".join(item_str) + "\n"
                        f.write(item_str)
                    print("temps execution", time.time() - time1)

                ###########################


                    # time.sleep(2)


                nlp_paragraph = nlp(paragraph['context'])
                list_ent_data = []
                for ent in nlp_paragraph.ents:
                    if ent.label_ in interesting_entities(type_question):
                        list_ent_data.append(normalize_answer(ent.text))

                dict_type_question[type_question][0] += 1

                if len(list_ent_data) == 0:
                    continue

                prediction = list_ent_data[0]
                ground_truths = list(map(lambda x: x['text'], question['answers']))
                # prediction = ground_truths[0]
                # print(ground_truths , prediction)
                # print(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
                # print(type_question)
                # print(dict_type_question[type_question][1])
                # time.sleep(5)

                dict_type_question[type_question][1] += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                dict_type_question[type_question][2] += metric_max_over_ground_truths(f1_score, prediction, ground_truths)





exact_match = f1 = moy_exact_match_all_types = moy_f1_all_types = total_nb_questions = 0

with open(path_dest + file_name + '.csv', 'w') as f:
    f.write("Question ; nb_de_questions ; exact_match ; f1"+ "\n")
    for key in dict_type_question:
        exact_match = 100.0 * (dict_type_question[key][1] / dict_type_question[key][0])
        f1 = 100.0 * dict_type_question[key][2] / dict_type_question[key][0]
        moy_exact_match_all_types += exact_match * dict_type_question[key][0]
        moy_f1_all_types += f1 * dict_type_question[key][0]
        total_nb_questions += dict_type_question[key][0]


        item_str = (key, str(dict_type_question[key][0]),str(exact_match), str(f1))
        item_str = " ; ".join(item_str) + "\n"
        f.write(item_str)
    item_str = ("MOYENNE TOUT TYPE", str(total_nb_questions), str(moy_exact_match_all_types/total_nb_questions ), str(moy_f1_all_types/total_nb_questions))
    item_str = " ; ".join(item_str) + "\n"
    f.write(item_str)
print("temps execution", time.time()-time1)
