import json
import sys
import time
import operator
path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_perso/'

sys.path.append("../../")
from nltk import sent_tokenize
from nltk import word_tokenize

dict = {}

#programme qui retourne la position (en mots) du span relative dans le texte

dict_phrase_position = {}
len_final = 0
nb_par = 0
def numeral_position(position, text, bool_normalize = True):
    position_word_span = len(word_tokenize(text[:position])) +1
    total_text_words = len(word_tokenize(text))
    return position_word_span/ max(float(total_text_words) * bool_normalize,1.0)


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
                    sentence_positionRelative = numeral_position(pos, paragraph['context'], True)
                    if not(sentence_positionRelative in list_ans):
                        total_answer += 1
                        list_ans.append(sentence_positionRelative)
                        sentence_positionRelative = round(sentence_positionRelative, 1)
                        if sentence_positionRelative in dict:
                            dict[sentence_positionRelative] += 1
                        else:
                            dict[sentence_positionRelative] = 1


list_dict_trie = sorted(dict.items(), reverse=False, key=operator.itemgetter(0))

with open('../position_span_words_dev.csv', 'w') as f:
    item_str = "position relative"+","+"comptage relatif"+","+"comptage"+","+"class" + "\n"
    f.write(item_str)
    for item in list_dict_trie:
        item_str = str(round(max(item[0]-0.05,0.0),2))+"-"+str(round(min(item[0]+0.05,1.0),2))+","+ str(item[1] / float(total_answer))+","+ str(item[1] )+",1" + "\n"
        f.write(item_str)


list_dict_trie = sorted(dict_phrase_position.items(), reverse=False, key=operator.itemgetter(0))
