'''
Programme qui vise à réecrire les données pour retrouver les question et la phrase dans lequel le span réponse se trouve.
'''

import json
import sys
path_data = '../../data/'
path_dest = '../../data_perso/'

sys.path.append("../../")
from utils.get_numeral_sentence import numeral_sentence

out_json = {}

#on recupere le numero de la phrase ncontenant le span

with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                list_ans = []
                for answer in question['answers']:
                    pos = answer['answer_start']
                    sentence_position = numeral_sentence(pos, paragraph['context'], False)
                    if not(sentence_position in list_ans) :
                        list_ans.append(sentence_position)

                # out_json[question['id']] = list_ans
                # out_json[question['id']] = [list_ans , question['question']]
                    out_json[question['id']] = answer["text"]
                    break

# with open(path_dest+'data_Dev.json', 'w') as outfile:
#     json.dump(out_json, outfile)
# with open(path_dest+'data_Dev_withQ.json', 'w') as outfile:
#     json.dump(out_json, outfile)
with open(path_dest + 'data_Dev_Parfait.json', 'w') as outfile:
    json.dump(out_json, outfile)