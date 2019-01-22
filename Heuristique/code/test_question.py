'''
Detection des spans réponse à l'aide des entité nommée
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import json
import sys
import time
import spacy


path_data = '../../../Data_Maitrise/data_separée_par_question/'
path_dest = '../data_results/'
selected_data = "dev"
# selected_data = "train"

# type_question = 1 #'Where?'
# type_question = 2 #'How much / many?'
# type_question = 3 #'What name / is called?'
# type_question = 4 #'Who?'
# type_question = 5 #'When / What year?'

type_question = 2

print("type de la question", type_question)
print()
if type_question == 1 :
    type_question_str = 'Where'
    interisting_entities = ("GPE", "LOC", "FAC", "ORG")
elif type_question == 2 :
    type_question_str = 'How_much'
    interisting_entities = ("MONEY","QUANTITY","PERCENT", "CARDINAL", "TIME","DATE", "ORDINAL")
elif type_question == 3:
    type_question_str = 'What_name'
    interisting_entities = ("PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE",'FAC')
elif type_question == 4:
    type_question_str = 'Who'
    interisting_entities = ("PERSON","ORG","NORP","GPE","PRODUCT")
elif type_question == 5:
    type_question_str = 'When'
    interisting_entities = ("TIME","DATE","EVENT")

time1 = time.time()
nlp = spacy.load('en_core_web_sm')


out_json = {}

f1 = exact_match = total_sentence = total_answer_in_ent = total_impossible_to_find_answer = 0
total_sentence_without_answer = 0
with open(path_data + 'data_question_'+type_question_str+'_'+selected_data+'.json', 'r') as input:
    d = json.load(input)
    input.close()
    list_ratio = []  # liste des rations de taille entitées / taille phrase
    for id in d:
        list_sentence_answer = d[id][0]
        list_answers = d[id][1]
        question = d[id][2]
        for sentence in list_sentence_answer:
            total_sentence += 1
            # if total_sentence % 100 == 0:
            #     print(100.0 * exact_match / total_sentence ,  100.0 * f1 / total_sentence)

            nlp_sentence = nlp(sentence)
            list_ent_data = []
            f1_list = []
            exact_match_list = []

            for ent in nlp_sentence.ents:
                if ent.label_ in interisting_entities:
                    list_ent_data.append(ent.text)
                    # exact_match += metric_max_over_ground_truths(exact_match_score, ent.text, list_answers)
                    # f1 += metric_max_over_ground_truths(f1_score, ent.text, list_answers)
                    exact_match_list.append(metric_max_over_ground_truths(exact_match_score, ent.text, list_answers))
                    f1_list.append(metric_max_over_ground_truths(f1_score, ent.text, list_answers))

            if len(list_ent_data) == 0:
                # print("reponse:   ", normalize_answer(answer) )
                # print(question)
                # print(sentence)
                # print(list_answers)
                # time.sleep(3)
                #
                total_sentence_without_answer += 1
                test_answer_inside_sentence = False
                for answer in list_answers:
                    if normalize_answer(answer) in normalize_answer(sentence):
                        test_answer_inside_sentence = True
                        break
                total_impossible_to_find_answer += not(test_answer_inside_sentence)
            else:
                exact_match += max(exact_match_list)
                f1 += max(f1_list)
                concatenation_ent = ' '.join(list_ent_data)
                for answer in list_answers:
                    if normalize_answer(answer) in normalize_answer(concatenation_ent) :
                        total_answer_in_ent += 1
                        break
                list_ratio.append(len(concatenation_ent) / float(len(sentence)))

                # print(d[id][2])
                # print(sentence)
                # for ent in nlp_sentence.ents:
                #     print(ent.text, ent.label_)
                # time.sleep(2)

print(100.0 * exact_match / total_sentence,  100.0 * f1 / total_sentence)
print(100.0 * total_sentence_without_answer / total_sentence)
print(100.0 * total_impossible_to_find_answer / total_sentence)
print(100.0 * total_answer_in_ent / total_sentence)
print("ratio de la taille", 100.0 * np.mean(list_ratio))
print("temps execution", time.time()-time1)
