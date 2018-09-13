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

model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../data/'
path_dest = '../data_test/'
time0 = time.time()
time1 = time.time()

with_steming_param = False
k_best_sentences = 1
n_param = 30
similarity_type = 1 #1 : cosine, 2:dice

dict_question={}
# replis sur tout les n et sur la phrase
def get_best_span_repli_all_less_n(model: fastText,id_question, list_sentence, question, k=1, n = 1,dict_question={}):
    vect_avg_question = avg_sentence_vector(question, model)
    grams_list = []
    for sentence in list_sentence:
        grams_list += list(ngrams(sentence.split(), n))
    if len(grams_list) == 0:
        return ""
    else:
        for gram in grams_list:
            gram_join = " ".join(gram)
            vect_avg_gram = avg_sentence_vector(gram_join, model)
            similarity = similarity_function(vect_avg_question, vect_avg_gram)
            if similarity > dict_question[id_question][0]:
                dict_question[id_question][0] = similarity
                dict_question[id_question][1] = gram_join
    return dict_question[id_question][1]


def similarity_function(vec1, vec2, similarity_type=similarity_type):
    if similarity_type == 1:
        return cosine_similarity(vec1, vec2)
    elif similarity_type == 2:
        return dice_similarity(vec1, vec2)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def dice_similarity(vec1, vec2):
    return (2.0 * np.dot(vec1, vec2)) / (np.dot(vec1, vec1) + np.dot(vec2, vec2))


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
class_final = "repli_phrase_n_cos"

with open(path_data + 'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

dict_rep = {}
for n in range(1, n_param): #pour tout les n
    dict_rep[n] = 0

nb_question = 0
out_json = {}
dict_question = {}
for data in d['data']:
    for paragraph in data['paragraphs']:
        for question in paragraph['qas']:
            nb_question += 1
            dict_question[question['id']] = [0,0]
            if nb_question % 1000 == 0:
                print(nb_question)
                print("temps :", time.time() - time1)
                time1 = time.time()
            taille_attendu = len(question['answers'][0]['text'].split())
            out_json[question['id']] = get_best_span_repli_all_less_n(model, question['id'], sent_tokenize(paragraph['context']), question['question'], k_best_sentences, taille_attendu, dict_question=dict_question)

with open(path_dest + 'data_toTest_fasText_truque.json', 'w') as outfile:
    json.dump(out_json, outfile)
    print(n)
    nb_question = 0



