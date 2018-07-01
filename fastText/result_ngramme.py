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

model = fastText.load_model('../wiki.simple/wiki.simple.bin')
#model = fastText.load_model('../embeding_perso_fastText/data_embeding.bin')
#model = fastText.load_model('../embeding_perso_fastText/train_steam_embeding.bin')

path_data = '../data/'
path_dest = '../data_perso/'

time1 = time.time()


with_steming_param = False
k_best_sentences = 1
n = 8
similarity_type = 1 #1 : cosine, 2:dice


def get_best_sentence_no_repli(model: fastText, list_sentence, question, k=1, n = 1):
    dictionnary = {}
    vect_avg_question = avg_sentence_vector(question, model)
    num_sentence = 1
    for sentence in list_sentence:
        grams_list = list(ngrams(sentence.split(), n))
        dictionnary[num_sentence] = 0
        for gram in grams_list:
            vect_avg_gram = avg_sentence_vector(" ".join(gram), model)
            dictionnary[num_sentence] = max(similarity_function(vect_avg_question, vect_avg_gram),dictionnary[num_sentence])
        num_sentence += 1
    dico_trie = sorted(dictionnary.items(), reverse=True, key=lambda t: t[1])
    list_return = list(map(lambda t: t[0], dico_trie))
    return list_return[:k]


# replis sur tout les n et sur la phrase
# def get_best_repli_sentence_and_all_less_n(model: fastText, list_sentence, question, k=1, n = 1):
#     dictionnary = {}
#     vect_avg_question = avg_sentence_vector(question, model)
#     num_sentence = 1
#     for sentence in list_sentence:
#         grams_list = []
#         for n in range(1,n+1):
#             grams_list += list(ngrams(sentence.split(), n))
#         dictionnary[num_sentence] = 0
#         for gram in grams_list:
#             vect_avg_gram = avg_sentence_vector(" ".join(gram), model)
#             similarity = similarity_function(vect_avg_question, vect_avg_gram)
#             dictionnary[num_sentence] = max(similarity, dictionnary[num_sentence])
#         if len(list(grams_list)) == 0 :
#             vect_avg_sentence = avg_sentence_vector(sentence, model)
#             dictionnary[num_sentence] = similarity_function(vect_avg_question, vect_avg_sentence)
#         num_sentence += 1
#     dico_trie = sorted(dictionnary.items(), reverse=True, key=lambda t: t[1])
#     list_return = list(map(lambda t: t[0], dico_trie))
#     return list_return[:k]


dict_question={}
# replis sur tout les n et sur la phrase
def get_best_repli_sentence_and_all_less_n(model: fastText,id_question, list_sentence, question, k=1, n = 1):
    vect_avg_question = avg_sentence_vector(question, model)
    num_sentence = 1
    global dict_question
    for sentence in list_sentence:
        grams_list = list(ngrams(sentence.split(), n))
        vect_avg_sentence = avg_sentence_vector(sentence, model)
        if id_question in dict_question :
            if not num_sentence in dict_question[id_question]:
                dict_question[id_question][num_sentence] = similarity_function(vect_avg_question, vect_avg_sentence)
        else:
            dict_question[id_question] = {num_sentence: similarity_function(vect_avg_question, vect_avg_sentence)}
        for gram in grams_list:
            vect_avg_gram = avg_sentence_vector(" ".join(gram), model)
            similarity = similarity_function(vect_avg_question, vect_avg_gram)
            dict_question[id_question][num_sentence] = max(similarity, dict_question[id_question][num_sentence])
        num_sentence += 1
    dico_trie = sorted(dict_question[id_question].items(), reverse=True, key=lambda t: t[1])
    list_return = list(map(lambda t: t[0], dico_trie))
    return list_return[:k]

# replis phrase
def get_best_repli_sentence_max(model: fastText, list_sentence, question, k=1, n = 1):
    dictionnary = {}
    vect_avg_question = avg_sentence_vector(question, model)
    num_sentence = 1
    for sentence in list_sentence:
        grams_list = list(ngrams(sentence.split(), n))
        dictionnary[num_sentence] = 0
        for gram in grams_list:
            vect_avg_gram = avg_sentence_vector(" ".join(gram), model)
            similarity = similarity_function(vect_avg_question, vect_avg_gram)
            dictionnary[num_sentence] = max(similarity, dictionnary[num_sentence])
        if len(grams_list) == 0:  # replis phrase
            vect_avg_sentence = avg_sentence_vector(sentence, model)
            dictionnary[num_sentence] = similarity_function(vect_avg_question, vect_avg_sentence)
        num_sentence += 1
    dico_trie = sorted(dictionnary.items(), reverse=True, key=lambda t: t[1])
    list_return = list(map(lambda t: t[0], dico_trie))
    return list_return[:k]


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
with open('resultat_ngrams'+class_final+'.csv', 'a') as f:
    for n in range(1,30): #pour tout les n
        print("n : ",n)
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
                        # list_ans = get_best_sentence_no_repli(model, sent_tokenize(paragraph['context']), question['question'], k_best_sentences, n)
                        list_ans = get_best_repli_sentence_and_all_less_n(model,question['id'], sent_tokenize(paragraph['context']), question['question'], k_best_sentences, n)
                        # list_ans = get_best_sentence_max(model, sent_tokenize(paragraph['context']),question['question'], k_best_sentences, n)
                        out_json[question['id']] = list_ans

        # with open(path_dest + 'data_toTest_fasText.json', 'w') as outfile:
        #     json.dump(out_json, outfile)


        with open(path_dest + 'data_Dev.json', 'r') as input:
            data_ref = json.load(input)
            input.close()

        # with open(path_dest + 'data_toTest_fasText.json', 'r') as input:
        #     data_to_evaluate = json.load(input)
        #     input.close()

        data_to_evaluate = out_json

        sum = 0
        s = 0
        for id in data_ref:
            j = 1
            # if s %1000 ==0 : print(s)
            s+=1
            for position in data_ref[id]:
                j += 1
                if position in data_to_evaluate[id]:
                    sum += 1
                    break
        print(sum / float(len(data_ref)))
        item_str = str(n)+";"+str((sum / float(len(data_ref)))) + ";" + class_final + "\n"
        f.write(item_str)
        print("temps :", time.time() - time1)

print(class_final)


