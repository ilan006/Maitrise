'''
programme qui retourne la position (en phrases) du span relative dans le texte
'''

import json
import operator
path_data = '../../../Data_Maitrise/data/'
path_dest = '../graph/'

from nltk import sent_tokenize

dict = {}

dict_phrase_position = {}
len_final = 0
nb_par = 0
def numeral_sentence_position(position, text, bool_normalize = True):
    '''
    Fonction qui retourne la position (en phrase) du span dans le texte
    :param position: la position du span dans le texte en char
    :param text: le paragraphe
    :param bool_normalize: boolean pour que la position retournee soit relative par rapport a la taille du texte
    :return: la position du span (en phrases)
    '''
    position_phrase_span = len(sent_tokenize(text[:position]))
    total_text_phrase = len(sent_tokenize(text))
    return position_phrase_span/ max(float(total_text_phrase) * bool_normalize, 1.0)


total_answer = 0
with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                list_ans = []
                for answer in question['answers']:
                    pos = answer['answer_start']
                    total_text_char = 0
                    do = True
                    for word in paragraph['context'].split():
                        total_text_char += len(word)+1
                        if total_text_char > pos and do:
                            do = False
                    sentence_positionRelative = numeral_sentence_position(pos, paragraph['context'], True)
                    if not(sentence_positionRelative in list_ans):
                        total_answer += 1
                        list_ans.append(sentence_positionRelative)
                        sentence_positionRelative = round(sentence_positionRelative, 1)
                        if sentence_positionRelative in dict:
                            dict[sentence_positionRelative] += 1
                        else:
                            dict[sentence_positionRelative] = 1

list_dict_trie = sorted(dict.items(), reverse=False, key=operator.itemgetter(0))

with open(path_dest + 'position_span_phrases_dev.csv', 'w') as f:
    item_str = "position relative"+","+"comptage relatif"+","+"comptage"+","+"class" + "\n"
    f.write(item_str)
    for item in list_dict_trie:
        item_str = str(round(max(item[0]-0.05,0.0),2))+"-" + str(round(min(item[0]+0.05,1.0),2))+","+ str(item[1] / float(total_answer))+","+ str(item[1] )+",1" + "\n"
        f.write(item_str)

list_dict_trie = sorted(dict_phrase_position.items(), reverse=False, key=operator.itemgetter(0))
