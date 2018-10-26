'''
ce systeme va aligner chaque mot de la question avec chaque phrase et va retourner le meilleur span contigu. Deux mots de la question ne peuvent pas etre associe au meme mot de question
'''
import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *
import operator

import nltk
from nltk import sent_tokenize,word_tokenize
from nltk import ngrams
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
born = 0.05

def align_question_sentence(question, sequence, lower_case_bool=True, born_min_align = 0 , print_align = False): # To do faire de manière recursive
    '''
    Fonction qui va aligner une question avec une phrase en prenant les cosine maximum tour a tour.
    '''

    list_words_question = word_tokenize(question.lower() if lower_case_bool else question)
    list_words_sequence = word_tokenize(sequence.lower() if lower_case_bool else sequence)

    nb_words_question = len(list_words_question)
    nb_words_sequence = len(list_words_sequence)

    vects_words_question = list(map(model.get_word_vector, list_words_question))
    vects_words_sentence = list(map(model.get_word_vector, list_words_sequence))

    edges = {} #Dictionnaire de similarité clef:[numero du mot dans la question , numero du mot dans la phrase], value : sim 2 mots de la question peuvent aligner sur le meme mot de la phrase
    edges_align =[]
    bool_words_list_question = [False] * nb_words_question
    bool_words_list_sequence = [False] * nb_words_sequence
    list_spans = []
    for i in range(0, nb_words_question):  # mots de la question
        for j in range(0, nb_words_sequence):
            sim = cosine_similarity(vects_words_question[i], vects_words_sentence[j])
            if sim>born_min_align:
                edges[i, j] = sim
    edges_sorted = sorted(edges.items(), key=operator.itemgetter(1), reverse=True)
    for elem in edges_sorted:
        num_word_question = elem[0][0]
        num_word_sequence = elem[0][1]
        if not(bool_words_list_question[num_word_question] or bool_words_list_sequence[num_word_sequence]): #Si le mot de la question n'a pas deja ete aligne
            edges_align.append(elem)
            bool_words_list_question[num_word_question] = True
            bool_words_list_sequence[num_word_sequence] = True

    span = []
    for i in range(0, nb_words_sequence):
        word_sequence = list_words_sequence[i] #mot de la phrase etudie
        if not (bool_words_list_sequence[i]):
            span.append(word_sequence)
        else:
            if not (len(span) == 0):
                list_spans.append(span)
            span = []
    if not (len(span) == 0):
        list_spans.append(span)

    list_final_spans = []
    for span in list_spans:
        for i in range(1,len(span)+1):
            list_final_spans += list(ngrams(span,i))

    if print_align: #on affiche l'alignement
        # print(question)
        # print(sequence)
        # print(edges_align)

        print("Alignement_graph_bipartite( \"" + question + "\",\"" + sequence + "\","+ str(edges_align) + ")")
        print("affiche_table_cosine(\"" +question+ "\",\"" +sequence + "\")")
        print("question=\"", question,"\"")
        print("sequence=\"", sequence,"\"")
        print("print(\"question = \", question)")
        print("print(\"sequence = \",sequence)")
    return list(map(lambda x : ' '.join(x),list_final_spans))

# print(list(map(lambda x : ' '.join(x),align_question_sentence("who are you Ilan?","I'm Ilan and you who ilan  you  I'm jean and you are."))))/

out_json = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                print_bool = False
                if num_quest % 1000 == 0 :
                    print(num_quest)
                    print_bool = True
                list_spans = []
                # for sentence in sent_tokenize(paragraph['context']):
                #     list_spans += align_question_sentence(question['question'], sentence)
                list_ans = get_best_sentence(model, sent_tokenize(paragraph['context']), question['question'], k_best_sentences)
                best_phrase = sent_tokenize(paragraph['context'])[list_ans[0]]
                list_spans = align_question_sentence(question['question'], best_phrase, born, print_align=print_bool)
                list_ans = []
                # print("list",list_spans)
                list_ans = get_best_sentence(model, list_spans, question['question'], k_best_sentences)
                try:
                    best_span = list_spans[list_ans[0]]
                except:
                    best_span =""

                out_json[question['id']] = best_span
                if print_bool :
                    print("print(\"span output:", out_json[question['id']], "\")")
                    print("print(\"reponses attendu:", question['answers'], "\")")
                # print(best_span)

                # print("question :   ", question['question'])
                # print("meilleure phrase :   ",best_phrase)
                # print("span retourné :   ", out_json[question['id']])
                # print("reponses atendues :   ", question['answers'])
                # print("    ")
                # if num_quest % 200 == 0:
                #     print("meilleure phrase :   ", best_phrase)
                #     print("Alignement_graph_bipartite(\"",question['question'],"\",\"", best_phrase, "\",", "2)")
                #     time.sleep(5)

with open(path_dest + 'data_toTest_System_alignement3_'+str(born)+'.json', 'w') as outfile:
    json.dump(out_json, outfile)


print("temps :", time.time() - time1)
