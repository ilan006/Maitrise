import fastText
from nltk.stem import PorterStemmer
from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import json
import sys
import re
import string
path_data = '../data/'
path_dest = '../data_perso/'

ps = PorterStemmer()

model = fastText.load_model('../wiki.simple/wiki.simple.bin')

path_data = '../data/'
path_dest = '../data_perso/'


def get_best_sentence_embeding(model : fastText, list_sentence,question):
    num_best_sentence = 1
    sim_best = 0.0
    vect_avg_question = avg_sentence_vector(question.split()[1:], model)

    num_sentence = 1
    for sentence in list_sentence :
        vect_avg_sentence = avg_sentence_vector(sentence.split(), model)
        similarity = cosine_similarity(vect_avg_question,vect_avg_sentence)
        if sim_best < similarity:
            sim_best = similarity
            num_best_sentence = num_sentence
        num_sentence += 1
    return num_best_sentence

def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)

def avg_sentence_vector(words, model : fastText, index2word_set = {}):
    num_features = model.get_dimension()
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    for word in words:
        featureVec = np.add(featureVec, model.get_word_vector(word))

    featureVec = np.divide(featureVec, max(len(words),1.0))
    return featureVec


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
    num_best_sentence = [1,1]
    sim_best = [0.0, 0.0]
    list_2best_sentence = ["",""]
    question_copy = question
    question = normalize_answer(question)
    num_sentence = 1
    for sentence in list_sentence:
        sentence_copy = sentence
        sentence = normalize_answer(sentence)
        list_word = set(map(ps.stem,sentence.split()))
        sim_match_word = 0
        for question_word in set(question.split()):
            sim_match_word += ps.stem(question_word) in list_word
        if sim_best[0] < sim_match_word:
            if sim_best[1] < sim_match_word:
                sim_best[0] = sim_best[1]
                num_best_sentence[0] = num_best_sentence[1]
                sim_best[1] = sim_match_word
                num_best_sentence[1] = num_sentence
                list_2best_sentence[0] = list_2best_sentence[1]
                list_2best_sentence[1] = sentence_copy
            else:
                sim_best[0] = sim_match_word
                num_best_sentence[0] = num_sentence
                list_2best_sentence[0] = sentence_copy
        num_sentence += 1
    return num_best_sentence[get_best_sentence_embeding(model,list_2best_sentence, question_copy)-1]

################################################################################################

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
                list_ans = get_best_sentence(sent_tokenize(paragraph['context']), question['question'])
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
        if position == data_to_evaluate[id]:
            sum += 1
            break

print(sum / float(len(data_ref)))

