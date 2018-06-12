import json
import sys
import time
import operator
path_data = '../../data/'
path_dest = '../../data_perso/'

sys.path.append("../../")
from nltk import sent_tokenize

dict = {}

#programme qui retourne la position (en mots) du span relative dans le texte


# retourne le numero de la phrase pour une position donné dans un text (en char)
def numeral_sentence(position, text, bool_normalize = True):
    tab_sent = sent_tokenize(text)
    taille = 0
    num_line = 1
    num_words = 0
    for sent in tab_sent:
        taille += len(sent) + 1
        num_words += sent.split()
        if position <= taille:
            return float(num_line) / max(len(tab_sent) * bool_normalize, 1)
        num_line += 1

# def numeral_position(position, text, bool_normalize = True):
#     total_text_words = 0
#     tab_sent = sent_tokenize(text)
#     taille = 0
#     num_line = 1
#     position_word_span = 0
#     do = True
#     for sent in tab_sent:
#         taille += len(sent) + 1
#         if position <= taille and do: #si on a trouver la ligne
#             position_word_span = total_text_words + 1
#             taille_mot = taille - (len(sent) + 1)
#             for word in sent.split():
#                 if position <= taille_mot:
#                     # print("taillle mot char",taille_mot )
#                     do = False
#                     break
#                 taille_mot += len(word) + 1
#                 position_word_span += 1
#         num_line += 1
#         total_text_words += len(sent.split())
#     return position_word_span/ max(float(total_text_words)*bool_normalize,1.0)

dict_phrase_position = {}
len_final = 0
nb_par = 0
def numeral_position(position, text, bool_normalize = True):
    global len_final,nb_par
    total_text_words = 0
    tab_sent = sent_tokenize(text)
    taille = 0
    num_line = 1
    position_word_span = 0
    do = True
    num_phrase = 0
    len_final += len(tab_sent[-1].split())/float(len(text.split())) #ratio de la taille(en mot) de la drniere phrase
    nb_par +=1
    for sent in tab_sent:
        num_phrase += 1
        taille += len(sent) + 1
        if position <= taille and do: #si on a trouver la phrase
            sentence_positionRelative = round(num_phrase/float(len(tab_sent)),1)
            if sentence_positionRelative in dict_phrase_position:
                dict_phrase_position[sentence_positionRelative] += 1
            else:
                dict_phrase_position[sentence_positionRelative] = 1
            if sentence_positionRelative == 1.0:
                pass
                # print("fin")
                # print(sent)
                # print(position)
                # print(len(text))

            position_word_span = total_text_words + 1
            taille_mot = taille - (len(sent) + 1)
            for word in sent.split():
                if position <= taille_mot:
                    # print("taillle mot char",taille_mot )
                    do = False
                    break
                taille_mot += len(word) + 1
                position_word_span += 1
        num_line += 1
        total_text_words += len(sent.split())
    return position_word_span/ max(float(total_text_words)*bool_normalize,1.0)


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
                        total_answer +=1
                        list_ans.append(sentence_positionRelative)
                        sentence_positionRelative = round(sentence_positionRelative, 1)
                        # sentence_positionRelative = sentence_positionRelative
                        if sentence_positionRelative in dict:
                            dict[sentence_positionRelative] += 1
                        else:
                            dict[sentence_positionRelative] = 1




# with open(path_dest+'data_Dev.json', 'w') as outfile:
#     json.dump(out_json, outfile)
# with open(path_dest+'data_Dev_withQ.json', 'w') as outfile:
#     json.dump(out_json, outfile)


list_dict_trie =sorted(dict.items(), reverse=False, key=operator.itemgetter(0))

with open('../position_span_words_dev.csv', 'w') as f:
    item_str = "position relative"+","+"comptage relatif"+","+"comptage"+","+"class" + "\n"
    f.write(item_str)
    for item in list_dict_trie:
        item_str = str(round(max(item[0]-0.05,0.0),2))+"-"+str(round(min(item[0]+0.05,1.0),2))+","+ str(item[1] / float(total_answer))+","+ str(item[1] )+",1" + "\n"
        # item_str = str(round(item[0], 2)) + "," + str(item[1] / float(total_answer)) + "," + str(item[1]) + ",1" + "\n"
        # item_str = str(round(item[0],4)) + "," + str(item[1] / float(total_answer)) + "," + str(item[1]) + ",1" + "\n"
        for k in range(item[1]):
            f.write(item_str)


list_dict_trie = sorted(dict_phrase_position.items(), reverse=False, key=operator.itemgetter(0))

# with open('../../extrac_sentence/data_perso/freq_position_train.csv', 'w') as f:
#     item_str = "position relative"+","+"comptage relatif"+","+"comptage"+","+"class" + "\n"
#     f.write(item_str)
#     for item in list_dict_trie:
#         item_str = str(round(max(item[0]-0.05,0.0),2))+"-"+str(round(min(item[0]+0.05,1.0),2))+","+ str(item[1] / float(total_answer))+","+ str(item[1] )+",1" + "\n"
#         f.write(item_str)
#
# print(float(len_final)/nb_par)