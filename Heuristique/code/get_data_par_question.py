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
# selected_data = "train"

# type_question ='When / What year?'
type_question = 'What name / is called?'
type_question_str = 'What_name'


# type_question = 1 #'Where?'
# type_question = 2 #'How much / many?'
# type_question = 3 #'What name / is called?'
# type_question = 4 #'Who?'
# type_question = 5 #'When / What year?'
type_question = 2

print("type de la question", type_question)
print()
if type_question == 1:
    type_question = 'Where?'
    type_question_str = 'Where'
elif type_question == 2:
    type_question = 'How much / many?'
    type_question_str = 'How_much'
elif type_question == 3:
    type_question = 'What name / is called?'
    type_question_str = 'What_name'
elif type_question == 4:
    type_question = 'Who?'
    type_question_str = 'Who'
elif type_question == 5:
    type_question = 'When / What year?'
    type_question_str = 'When'


time1 = time.time()


out_json = {}

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
                    sentence_position = num_sentence(pos, answer["text"], paragraph['context'])
                    if not(sentence_position in list_ans):
                        list_ans.append(sentence_position)
                        list_sentence_answer.append(sent_tokenize(paragraph['context'])[sentence_position])
                        list_answers.append(answer["text"])
                out_json[question['id']] = [list_sentence_answer,list_answers, question['question'] ]


with open(path_dest + 'data_question_'+type_question_str+'_'+selected_data+'.json', 'w') as outfile:
    json.dump(out_json, outfile)
print(time.time() - time1)