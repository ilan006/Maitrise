"""
fonctions permettant de trouver la position des repponses dans le texte.
"""
from nltk import sent_tokenize
from evaluate import *
import time

def num_sentence(position, answer_text, text):
    '''
    fonction retournant pour une position donnée (en char) le numero de la phrase correspondante
    :param position: position en char
    :param answer_text: la reponse sous forme d'une string
    :param text: le paragraphe contenant l'ensmeble des phrases
    :return: le numéro de la phrase correspondant à la position donnée
    '''
    tab_sent = sent_tokenize(text[:position])
    tab_all_sentences = sent_tokenize(text)
    if len(tab_sent) == 0:
        # si la reponse se trouve en début de paragraghe
        retours = 0
    else:
        if tab_sent[-1] in tab_all_sentences:
            # si la derniere phrase est complete on retourne la phrase suivante
            retours = len(tab_sent)
        else:
            retours = len(tab_sent)-1

    try :
        if normalize_answer(answer_text) in normalize_answer(tab_all_sentences[retours]):
            return retours
        else :
            return min(retours+1, len(tab_all_sentences)-1)
    except:
        print("erreur de detection de phrase")
        print(retours , len(tab_all_sentences))
        time.sleep(5)
        return 0

def get_sentence(answer, text):
    '''
    Fonction qui va retourner la phrase (string) qui inclu la réponse
    :param answer: la question incluant sa position et son text
    :param text: le paragraphe
    :return:la phrase incluant la reponse
    '''
    index_sentence = num_sentence(answer['answer_start'], answer["text"], text)
    return sent_tokenize(text)[index_sentence]


# def num_sentence(position, text):
#     '''
#     fonction retournant pour une position donnée (en char) le numero de la phrase correspondante
#     :param position: position en char
#     :param text: le paragraphe contenant l'ensmeble des phrases
#     :return: le numéro de la phrase correspondant à la position donnée
#     '''
#     tab_sent = sent_tokenize(text[:position])
#     tab_all_sentences = sent_tokenize(text)
#     if len(tab_sent) == 0 :
#         return 0
#     else:
#         if tab_sent[-1] in tab_all_sentences:
#             return len(tab_sent)
#         else:
#             return len(tab_sent)-1
#     # return max(len(tab_sent)-1,0)


def pos_relative_sentence(position, text, bool_normalize = True):
    '''
    :param position : la position en char determinant la phrase d'interet
    :param text : le text contenant la phrase d'interet
    :param bool_normalize: booleen indiquant si il faut normaliser la position par la taille du texte
    :return: la postion relative de la phrase dans le paragraphe
    '''
    tab_sent = sent_tokenize(text)
    taille = 0
    num_line = 1
    for sent in tab_sent:
        taille += len(sent) + 1
        if position <= taille:
            return float(num_line) / max(len(tab_sent) * bool_normalize, 1)
        num_line += 1