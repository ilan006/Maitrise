import fastText
from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import json
import sys
import time
from nltk import ngrams
from nltk.stem import PorterStemmer
ps = PorterStemmer()
path_data = '../../data/'
path_dest = '../../data_perso/'

sys.path.append("../../")
from nltk import sent_tokenize
from nltk import word_tokenize
import operator

dict = {}

#programme qui retourne la position (en mots) du span relative dans le texte

dict_phrase_position = {}
len_final = 0
nb_par = 0



model = fastText.load_model('../../wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../../data/'
path_dest = '../../data_perso/'

time0 = time.time()
time1 = time.time()

with_steming_param = False
k_best_sentences = 1
n_param = 30


def avg_sentence_vector(sentence, model: fastText, with_steming = with_steming_param):
    num_features = model.get_dimension()
    featureVec = np.zeros(num_features, dtype="float32")
    words = word_tokenize(sentence)
    print(words)
    words = (list(map(ps.stem, words)) if with_steming else words)
    for word in words:
        featureVec = np.add(featureVec, model.get_word_vector(word))
    featureVec = np.divide(featureVec, float(max(len(words),1)))
    return featureVec


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def numeral_position(position, text, bool_normalize = True):
#     global len_final,nb_par
#     total_text_words = 0
#     tab_sent = sent_tokenize(text)
#     taille = 0
#     num_line = 1
#     position_word_span = 0
#     do = True
#     num_phrase = 0
#     nb_par +=1
#     for sent in tab_sent:
#         num_phrase += 1
#         taille += len(sent) + 1
#         if position <= taille and do: #si on a trouver la phrase
#                 # print("fin")
#                 # print(sent)
#                 # print(position)
#                 # print(len(text))
#
#             position_word_span = total_text_words + 1
#             taille_mot = taille - (len(sent) + 1)
#             for word in word_tokenize(sent):
#                 if position <= taille_mot:
#                     do = False
#                     break
#                 taille_mot += len(word) + 1
#                 position_word_span += 1
#         num_line += 1
#         total_text_words += len(sent.split())
#     # print("taille du text du text trouver en haut   ", len(text.split()))
#     # print("taille du text du text trouver en haut   ", text.split())
#     return position_word_span



def numeral_position(position, text, bool_normalize = True):
    tab_sent = sent_tokenize(text)
    taille = -1
    position_word_span = 1
    for sent in tab_sent:
        taille += len(sent) + 1
        if position <= taille: #si on a trouver la phrase
            taille_mot = taille - (len(sent) + 1)
            sent = sent.replace(" ", "#")
            print(word_tokenize(sent))
            for word in word_tokenize(sent):
                taille_mot += len(word)
                if position <= taille_mot:
                    print("mot correspondant", word)
                    return position_word_span
                if word != "#":
                    position_word_span += 1
        else :
            position_word_span += len(word_tokenize(sent))


total_answer = 0
with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            text_parag_tokens_1 = sent_tokenize(paragraph['context'])
            text_parag_tokens_1 = list(map(lambda s: s.split(), text_parag_tokens_1))
            text_parag_tokens = []
            for list_words in text_parag_tokens_1:
                text_parag_tokens += list_words
            # print(text_parag_tokens)
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
                    position_Word = numeral_position(pos, paragraph['context'], True)
                    if not(sentence_positionRelative in list_ans):
                        print("debut")
                        print("reponse attendu   ", answer['text'])
                        # print("pos", pos)
                        # print("lettre attendu   ", paragraph['context'][pos])
                        # print("taille du texte   ", len(text_parag_tokens))
                        # print("position   ", position_Word)
                        # print("reponse sorti   ", text_parag_tokens[position_Word-1])
                        # print("FIn")
                        total_answer += 1
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