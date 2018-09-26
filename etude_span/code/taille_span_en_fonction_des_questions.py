'''
programme qui retourne dans un fichier la distribution des tailles des span-reponses en fonction du type de question
'''


import json
import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *


path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'

dict = {}
nb_answers = 0
total_words = 0

with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                list_ans = []
                for answer in question['answers']:
                    span = answer['text']
                    if span not in list_ans:
                        list_ans.append(span)
                        nb_words_ans = len(span.split())
                        total_words += nb_words_ans
                        nb_answers += 1
                        type_question = ClassifyQuestion(question['question'])
                        if type_question in dict:
                            dict[type_question] ["total_ans"] += 1
                            if nb_words_ans in dict[type_question]:
                                dict[type_question][nb_words_ans] += 1
                            else:
                                dict[type_question][nb_words_ans] = 1
                        else:
                            dict[type_question] = {nb_words_ans: 1}
                            dict[type_question] = {"total_ans": 1}
                if nb_answers % 1000 == 0:
                    print(nb_answers)
print('nombre de span diffrent :', nb_answers)
print('taille moyen du span-reponse :', float(total_words)/nb_answers)

with open(path_dest + 'taille_span_dev_en_fonction_des_questions.csv', 'w') as f:
    item_str = "type question" + ',' + "taille" + "," + "compte relatif" + "," + "compte" + "\n"
    f.write(item_str)
    for type_question in dict:
        for taille in dict[type_question]:
            if taille == "total_ans":
                continue
            item_str = str(type_question) + "," + str(taille) + "," + str(dict[type_question][taille]/ float(dict[type_question]["total_ans"])) +"," +str(dict[type_question][taille]) + "\n"
            f.write(item_str)