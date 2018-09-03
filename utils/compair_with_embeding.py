from nltk import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
from nltk.stem import PorterStemmer

ps = PorterStemmer()


# Fonction qui va retourner le numero de la meilleure phrase
def get_best_sentence(model: Word2Vec, list_sentence, question, with_steming=True):
    num_best_sentence = 1
    sim_best = 0.0
    vect_avg_question = avg_sentence_vector(question, model)

    num_sentence = 1
    for sentence in list_sentence:
        vect_avg_sentence = avg_sentence_vector(sentence, model, with_steming)
        similarity = cosine_similarity(vect_avg_question, vect_avg_sentence)
        if sim_best < similarity:
            sim_best = similarity
            num_best_sentence = num_sentence
        num_sentence += 1
    return num_best_sentence


# Fonction qui retourne la similarite cosinus
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Fonction qui retourne le vecteur moyen d'embeding d'une phrase
def avg_sentence_vector(sentence, model: Word2Vec, with_steming=True):
    num_features = model.vector_size
    featureVec = np.zeros((num_features,), dtype="float32")
    words = sentence.split()
    words = (list(map(ps.stem, words)) if with_steming else words)
    for word in words:
        if word in model.wv.vocab:
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, float(max(len(words), 1)))
    return featureVec