import fastText
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




def get_best_sentence(list_sentence,question):
    num_best_sentence = 1
    sim_best = 0
    question = normalize_answer(question)
    num_sentence = 1
    for sentence in list_sentence:
        sentence = normalize_answer(sentence)
        list_word = set(map(ps.stem,sentence.split()))
        sim_match_word = 0
        for question_word in set(question.split()):
            sim_match_word += ps.stem(question_word) in list_word
        if sim_best < sim_match_word:
            sim_best = sim_match_word
            num_best_sentence = num_sentence
        num_sentence += 1
    return num_best_sentence


##########################################################################

out_json = {}
with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                # if (num_quest % 1000) == 0: print(num_quest)
                num_quest += 1
                list_ans = []
                list_ans = [get_best_sentence(sent_tokenize(paragraph['context']), question['question'])]
                out_json[question['id']] = list_ans

with open(path_dest + 'data_toTest.json', 'w') as outfile:
    json.dump(out_json, outfile)


with open(path_dest + 'data_Dev.json', 'r') as input:
    data_ref = json.load(input)
    input.close()

with open(path_dest + 'data_toTest.json', 'r') as input:
    data_to_evaluate = json.load(input)
    input.close()

sum = 0
for id in data_ref:
    j = 1
    for position in data_ref[id]:
        j += 1
        if position == data_to_evaluate[id][0]:
            sum += 1
            break

print(sum / float(len(data_ref)))

print(time.time() - time1)

