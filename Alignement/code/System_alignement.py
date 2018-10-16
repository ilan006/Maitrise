'''
ce systeme va aligner chaque mot de la question avec chaque phrase et va retourner le meilleur span contigu.
'''
import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *
import operator

import nltk
from nltk import sent_tokenize,word_tokenize
import json
import time
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import fastText
model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')

#ignorer toutes les depreciations de fonctions
import warnings
warnings.simplefilter('ignore')


path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'


time1 = time.time()


with_steming_param = True
k_best_sentences = 1

with_steming = True


def align_question_sentence(question, sequence, lower_case_bool=True, born_min_align = 0):
    '''
    Fonction qui va aligner une question avec une phrase en prenant les cosine maximum tour a tour.
    '''

    list_words_question = word_tokenize(question.lower() if lower_case_bool else question)
    list_words_sequence = word_tokenize(sequence.lower() if lower_case_bool else sequence)

    nb_words_question = len(list_words_question)
    nb_words_sequence = len(list_words_sequence)

    vects_words_question = list(map(model.get_word_vector, list_words_question))
    vects_words_sentence = list(map(model.get_word_vector, list_words_sequence))

    edges = {} #Dictionnaire de similarité clef:[numero du mot dans la question , numero du mot dans la phrase], value : sim
    edges_align =[]
    bool_words_list_question = [False] * nb_words_question
    bool_words_list_sequence = [False] * nb_words_sequence
    list_spans = []
    for i in range(0, nb_words_question):  # mots de la question
        for j in range(0, nb_words_sequence):
            edges[i, j] = cosine_similarity(vects_words_question[i], vects_words_sentence[j])
            edges_sorted = sorted(edges.items(), key=operator.itemgetter(1), reverse=True)
    print(list(map(lambda x: x[0], edges_sorted)))
    for elem in edges_sorted:
        num_word_question = elem[0][0]
        num_word_sequence = elem[0][1]
        if not(bool_words_list_question[num_word_question]): #Si le mot de la question n'a pas deja ete aligne
            edges_align.append(elem)
            bool_words_list_question[num_word_question] = True
            bool_words_list_sequence[num_word_sequence] = True

    span = ""
    for i in range(0, nb_words_sequence):
        word_sequence = list_words_sequence[i]
        if not (bool_words_list_sequence[i]):
            span += word_sequence + " "
            list_spans.append(word_sequence)
            list_spans.append(span)
        else:
            span = ""
    print(edges_sorted)
    print(edges_align)
    print(list_spans)

# def align(question,sentence):
#     span =""
#     questions_words = nltk.word_tokenize(question)
#     # questions_words = (list(map(ps.stem, questions_words)) if with_steming else questions_words)
#     sentence_words = nltk.word_tokenize(sentence)
#     # sentence_words = (list(map(ps.stem, sentence_words)) if with_steming else sentence_words)
#
#
#
#     for sentence_word in sentence_words :
#         if not(sentence_word in questions_words):
#             span += sentence_word +" "
#         else:
#             questions_words.remove(sentence_word)
#     return span
#
# out_json = {}
# with open(path_data + 'dev-v1.1.json', 'r') as input:
#     d = json.load(input)
#     input.close()
#
#     num_quest = 1
#     for data in d['data']:
#         for paragraph in data['paragraphs']:
#             for question in paragraph['qas']:
#                 num_quest += 1
#                 if num_quest % 1000 == 0 :
#                     print(num_quest)
#                 list_ans = []
#                 list_ans = get_best_sentence(model, sent_tokenize(paragraph['context']), question['question'], k_best_sentences)
#                 best_phrase = sent_tokenize(paragraph['context'])[list_ans[0]-1]
#
#                 list_words = word_tokenize(best_phrase)
#                 out_json[question['id']] = align( question['question'], best_phrase)
#
#                 # print("question :   ", question['question'])
#                 # print("meilleure phrase :   ",best_phrase)
#                 # print("span retourné :   ", out_json[question['id']])
#                 # print("reponses atendues :   ", question['answers'])
#                 # print("    ")
#                 # if num_quest % 200 == 0:
#                 #     print("meilleure phrase :   ", best_phrase)
#                 #     print("Alignement_graph_bipartite(\"",question['question'],"\",\"", best_phrase, "\",", "2)")
#                 #     time.sleep(5)
#
# with open(path_dest + 'data_toTest_Alignement.json', 'w') as outfile:
#     json.dump(out_json, outfile)
#
#
# print("temps :", time.time() - time1)

align_question_sentence("who are you?","I'm Ilan.")