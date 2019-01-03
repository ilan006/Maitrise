'''
Programme qui vise à retrouver la phrase dans lequel le span réponse identifier a été extrait.
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
type_question ='When / What year?'
type_question_str = 'When'
time1 = time.time()


out_json = {}

#on recupere le numero de la phrase ncontenant le span
num_quest = 0
with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                if ClassifyQuestion(question['question']) != type_question:
                    continue
                num_quest += 1
                if num_quest %1000 ==0:
                    print(num_quest)
                    time.sleep(5)

                list_ans = []
                list_sentence_answer = []
                list_answers = []
                for answer in question['answers']:
                    pos = answer['answer_start']
                    sentence_position = num_sentence(pos, paragraph['context'])
                    if not(sentence_position in list_ans):
                        list_ans.append(sentence_position)
                        list_sentence_answer.append(sent_tokenize(paragraph['context'])[sentence_position])
                        list_answers.append(answer["text"])
                out_json[question['id']] = [list_sentence_answer,list_answers, question['question'] ]


with open(path_dest + 'data_question_'+type_question_str+'_'+selected_data+'.json', 'w') as outfile:
    json.dump(out_json, outfile)
