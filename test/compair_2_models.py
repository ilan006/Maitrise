from nltk.stem import PorterStemmer
from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import json
import sys
import string
import re
import corenlp.client as c
import corenlp

question = "When was the most recent Super Bowl hosted in the South Florida/Miami area?"

# We assume that you've downloaded Stanford CoreNLP and defined an environment
# variable $CORENLP_HOME that points to the unzipped directory.
# The code below will launch StanfordCoreNLPServer in the background
# and communicate with the server to annotate the sentence.
client = c.CoreNLPClient(annotators="tokenize ssplit pos lemma parse".split())



path_data = '../data/'
path_dest = '../data_perso/'

# 0 = 'Other'
# 1 = 'What name / is called?'
# 2 = 'When / What year?'
# 3 = 'What / Which NN[*]?'
# 4 = 'What VB[*]?'
# 5 = 'How much / many?'
# 6 = 'How?'
# 7 = 'Who?'
# 8 = 'Where?'

def ClassifyQuestion(question):
    question = client.annotate(question)
    tokens = question.sentence[0].token

    wh_token_index = None
    for i, token in enumerate(tokens):
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            wh_token_index = i
            break

    if wh_token_index is None:
        return 0

    edge_types = set()
    edge_lemmas = set()
    for edge in question.sentence[0].basicDependencies.edge:
        if edge.dep in ['punct', 'cop', 'root']:
            continue

        if edge.source == wh_token_index + 1:
            if tokens[edge.target - 1].lemma == 'in':
                continue
            edge_types.add(tokens[wh_token_index].lemma + ' - ' + edge.dep + ' -> ' + tokens[edge.target - 1].pos)
            edge_lemmas.add(tokens[wh_token_index].lemma + ' - ' + edge.dep + ' -> ' + tokens[edge.target - 1].lemma)

        if edge.target == wh_token_index + 1:
            edge_types.add(tokens[wh_token_index].lemma + ' <- ' + edge.dep + ' - ' + tokens[edge.source - 1].pos)
            edge_lemmas.add(tokens[wh_token_index].lemma + ' <- ' + edge.dep + ' - ' + tokens[edge.source - 1].lemma)

    if 'what <- det - name' in edge_lemmas or 'what - nsubj -> name' in edge_lemmas or 'what <- dobj - call' in edge_lemmas or 'what <- nsubjpass - call' in edge_lemmas:
        return 1

    if tokens[
        wh_token_index].lemma == 'when' or 'what <- det - year' in edge_lemmas or 'what - nsubj -> year' in edge_lemmas:
        return 2

    for wh_token_type in ['what', 'which']:
        for edge_type in [' <- det - ', ' - nsubj -> ', ' <- nsubj - ', ' <- dobj - ', ' - dep -> ', ' <- nmod - ']:
            for noun_type in ['NN', 'NNS', 'NNP']:
                if wh_token_type + edge_type + noun_type in edge_types:
                    return 3

    for edge_type in [' <- dobj - ', ' <- nsubjpass - ', ' <- nsubj - ', ' - nsubj -> ', ' - dep -> ']:
        for verb_type in ['VB', 'VBN', 'VBZ', 'VBP', 'VBD', 'VBG']:
            if 'what' + edge_type + verb_type in edge_types:
                return 4

    if 'how <- advmod - many' in edge_lemmas or 'how <- advmod - much' in edge_lemmas:
        return 5

    if tokens[wh_token_index].lemma == 'how':
        return 6

    if tokens[wh_token_index].lemma in ['who', 'whom', 'whose']:
        return 7

    if tokens[wh_token_index].lemma == 'where':
        return 8

    return 0

with open(path_dest + 'data_Dev_withQ.json', 'r') as input:
    data_ref = json.load(input)
    input.close()

with open(path_dest + 'data_toTest_ExactMatch.json', 'r') as input:
    data_toTest_ExactMatch = json.load(input)
    input.close()

with open(path_dest + 'data_toTest_fasText.json', 'r') as input:
    data_toTest_fasText = json.load(input)
    input.close()

sum = 0
total_good_just_fast = 0
total_good_just_exactMatch = 0
num_each_type_question = [0] * 9
num_fast = [0] * 9
num_exactMatch = [0] * 9
k = 0
for id in data_ref:
    j = 1
    num_type_question = ClassifyQuestion(data_ref[id][1])
    num_each_type_question[num_type_question] += 1
    k += 1
    if k % 1000 == 0: print(k)
    bool_list = [False] * 2
    for position in data_ref[id][0]:
        j += 1
        if position in data_toTest_ExactMatch[id]:
            num_exactMatch[num_type_question] += 1
            bool_list[0] = True
        if position in data_toTest_fasText[id]:
            num_fast[num_type_question] += 1
            bool_list[1] = True
        if bool_list[0] * bool_list[1] == 1:
            sum += 1
            break
        if bool_list[1] and not bool_list[0]: total_good_just_fast +=1
print(num_exactMatch)
print(num_fast)