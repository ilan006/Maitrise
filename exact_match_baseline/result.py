from nltk.stem import PorterStemmer
from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import json
import sys
import string
import re
import time
path_data = '../data/'
path_dest = '../data_perso/'

time1 = time.time()

ps = PorterStemmer()

#Params
with_steming_param = True
k_best_sentences = 1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_best_sentence(list_sentence,question, k=1, with_steming = with_steming_param):
    dictionnary = {}
    question_words = normalize_answer(question).split()
    question_words = set(map(ps.stem,question_words) if with_steming else question_words)
    num_sentence = 1
    for sentence in list_sentence:
        sentence = normalize_answer(sentence)
        list_word = set(map(ps.stem, sentence.split()) if with_steming else sentence.split() )
        sim_match_word = 0
        for word in question_words:
            sim_match_word += word in list_word
        dictionnary[num_sentence] = sim_match_word
        num_sentence += 1
    dico_trie = sorted(dictionnary.items(), reverse=True, key=lambda t: t[1])
    list_return = list(map(lambda t: t[0], dico_trie))
    return list_return[:k]


##########################################################################

out_json = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                num_quest += 1
                list_ans = []
                list_ans = get_best_sentence(sent_tokenize(paragraph['context']), question['question'], k = k_best_sentences)
                out_json[question['id']] = list_ans

with open(path_dest + 'data_toTest_ExactMatch.json', 'w') as outfile:
    json.dump(out_json, outfile)


with open(path_dest + 'data_Dev.json', 'r') as input:
    data_ref = json.load(input)
    input.close()

with open(path_dest + 'data_toTest_ExactMatch.json', 'r') as input:
    data_to_evaluate = json.load(input)
    input.close()

sum = 0
for id in data_ref:
    j = 1
    for position in data_ref[id]:
        j += 1
        if position in data_to_evaluate[id]:
            sum += 1
            break

print(sum / float(len(data_ref)))

print("temps :", time.time() - time1)

