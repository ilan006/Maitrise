# coding= utf-8

import os, sys
""" Official evaluation script for v1.1 of the SQuAD dataset. """
from collections import Counter
import string
import re
import argparse
import json
import sys
from gensim.models import Word2Vec
from nltk import sent_tokenize
sys.path.append("../../")
from utils.get_numeral_sentence import numeral_sentence
from utils.compair_with_embeding import get_best_sentence
import pickle
path_data = '../../data/'
path_dest = '../../data_perso/'



def creat_embeding( size=100, window=10, min_count=3, workers=16, sg=0):
    path_data = '../../data/'
    path_dest = '../../data_perso/'
    with open(path_dest + 'train_concat.txt', 'r') as f:
        input_text = f.read()

    # remove parenthesis
    input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
    # store as list of sentences
    tab_sent = sent_tokenize(input_text_noparens)

    sentences_segm = []
    for sent_str in tab_sent:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        tokens = sent_str.lower().split()
        sentences_segm.append(tokens)

    model = Word2Vec(sentences=sentences_segm, size=size, window=window, min_count=min_count, workers=workers, sg=sg)

    model.save(path_dest + 'embeding.txt')


def result():
    with open(path_dest + 'data_Dev.json', 'r') as input:
        data_ref = json.load(input)
        input.close()

    with open(path_dest + 'data_toTest.json', 'r') as input:
        data_to_evaluate = json.load(input)
        input.close()

    sum = 0
    for id in data_ref:
        for position in data_ref[id]:
            if position == data_to_evaluate[id][0]:
                sum += 1
                break

    return (sum / float(len(data_ref)))

def out_test() :
    model = Word2Vec.load(path_dest + 'embeding.txt')

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
                    list_ans = [get_best_sentence(model, sent_tokenize(paragraph['context']), question['question'])]
                    out_json[question['id']] = list_ans

    with open(path_dest + 'data_toTest.json', 'w') as outfile:
        json.dump(out_json, outfile)


best_param = []
best_ratio = 0.0
for size in range(1, 15):
        creat_embeding(50, size, min_count=3, workers=16, sg=0)
        out_test()
        ratio = result()
        if ratio > best_ratio:
            best_ratio = ratio
            best_param = [size]
            print(best_ratio)
            print(best_param)
print(best_param)