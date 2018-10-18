'''
ce systeme va retourne comme span-réponse la best-sentence moins les mots de la question
'''
import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import fastText
import nltk
from nltk import sent_tokenize,word_tokenize
import json
import time
from nltk.stem import PorterStemmer
ps = PorterStemmer()

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_to_test/'


time1 = time.time()


with_steming_param = True
k_best_sentences = 1

with_steming = True

def align(question,sentence):
    span =""
    questions_words = nltk.word_tokenize(question)
    # questions_words = (list(map(ps.stem, questions_words)) if with_steming else questions_words)
    sentence_words = nltk.word_tokenize(sentence)
    # sentence_words = (list(map(ps.stem, sentence_words)) if with_steming else sentence_words)
    for sentence_word in sentence_words :
        if not(sentence_word in questions_words):
            span += sentence_word +" "
        else:
            questions_words.remove(sentence_word)
    return span

out_json = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                if num_quest % 1000 == 0 :
                    print(num_quest)
                list_ans = []
                list_ans = get_best_sentence(model, sent_tokenize(paragraph['context']), question['question'], k_best_sentences)
                best_phrase = sent_tokenize(paragraph['context'])[list_ans[0]]

                list_words = word_tokenize(best_phrase)
                out_json[question['id']] = align( question['question'], best_phrase)

                # print("question :   ", question['question'])
                # print("meilleure phrase :   ",best_phrase)
                # print("span retourné :   ", out_json[question['id']])
                # print("reponses atendues :   ", question['answers'])
                # print("    ")
                # if num_quest % 200 == 0:
                #     print("meilleure phrase :   ", best_phrase)
                #     print("Alignement_graph_bipartite(\"",question['question'],"\",\"", best_phrase, "\",", "2)")
                #     time.sleep(5)

with open(path_dest + 'data_toTest_Alignement.json', 'w') as outfile:
    json.dump(out_json, outfile)


print("temps :", time.time() - time1)
