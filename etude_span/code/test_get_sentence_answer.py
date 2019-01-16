'''
Programme qui vise à tester si la phrase recuperer à l'aide de la position du span-réponse contient bien la réponse
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import json
import sys
import time


path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_separée_par_question/'
selected_data = "dev"
# selected_data = "train"
time1 = time.time()


num_quest = 0
total_answers = 0
total_problem = 0
with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                if num_quest % 1000 ==0:
                    print(num_quest)
                    # time.sleep(5)

                list_sentence_answer = []
                list_answers = []
                for answer in question['answers']:
                    total_answers += 1
                    pos = answer['answer_start']
                    sentence_position = num_sentence(pos, answer["text"], paragraph['context'])
                    sentence_answer = sent_tokenize(paragraph['context'])[sentence_position]
                    if not normalize_answer(answer["text"]) in normalize_answer(sentence_answer):
                        total_problem += 1

print(100.0 * total_problem/total_answers)
print("temps :", time.time() - time1)