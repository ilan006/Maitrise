'''
programme qui va retrouner un fichier jason de type id: [question, reponses]
'''
import json
import sys
path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_perso/'

out_json = {}

with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()
num_quest = 0
for data in d['data']:
    for paragraph in data['paragraphs']:
        for question in paragraph['qas']:
            num_quest += 1
            list_ans = []
            for answer in question['answers']:
                if not answer["text"] in list_ans:
                    list_ans.append(answer["text"])
                if (num_quest % 1000) == 0 : print(num_quest)
                pos = answer['answer_start']
            out_json[question['id']] = [question['question'], list_ans ]


#
with open(path_dest + 'data_Q_and_answer.json', 'w') as outfile:
    json.dump(out_json, outfile)