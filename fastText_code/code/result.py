import sys
sys.path.insert(0, '../..')

import fastText
from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import json
import sys
import time
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import utils

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_perso/'

time1 = time.time()


with_steming_param = False
k_best_sentences = 1

def get_best_sentence(model: fastText, list_sentence, question, k=1):
    dictionnary = {}
    vect_avg_question = avg_sentence_vector(question, model)
    num_sentence = 1
    for sentence in list_sentence:
        vect_avg_sentence = avg_sentence_vector(sentence, model)
        dictionnary[num_sentence] = cosine_similarity(vect_avg_question, vect_avg_sentence)
        num_sentence += 1
    dico_trie = sorted(dictionnary.items(), reverse=True, key=lambda t: t[1])
    list_return = list(map(lambda t: t[0], dico_trie))
    return list_return[:k]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def avg_sentence_vector(sentence, model: fastText, with_steming = with_steming_param):
    num_features = model.get_dimension()
    featureVec = np.zeros(num_features, dtype="float32")
    words = sentence.split()
    words = (list(map(ps.stem, words)) if with_steming else words)
    for word in words:
        featureVec = np.add(featureVec, model.get_word_vector(word))
    featureVec = np.divide(featureVec, float(max(len(words),1)))
    return featureVec

################################################################################################

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
                out_json[question['id']] = list_ans

with open(path_dest + 'data_toTest_fasText.json', 'w') as outfile:
    json.dump(out_json, outfile)


with open(path_dest + 'data_Dev.json', 'r') as input:
    data_ref = json.load(input)
    input.close()

with open(path_dest + 'data_toTest_fasText.json', 'r') as input:
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


