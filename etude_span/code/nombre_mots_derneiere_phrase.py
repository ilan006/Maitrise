import json
import sys
import time
import operator
path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_perso/'

sys.path.append("../../")
from nltk import sent_tokenize
from nltk import word_tokenize

dict = {}

#programme qui retourne la position (en mots) du span relative dans le texte

dict_phrase_position = {}
len_final = 0
nb_par = 0
def numeral_position(position, text, bool_normalize = True):
    position_phrase_span = len(sent_tokenize(text[:position]))
    total_text_phrase = len(sent_tokenize(text))
    return position_phrase_span/ float(total_text_phrase)


total_answer = 0
nb_phrase_total = 0
nb_paragraphe = 0
nb_total_mots = 0
nb_words_phrase_final = 0
with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            nb_paragraphe += 1
            nb_phrase_total += len(sent_tokenize(paragraph['context']))
            nb_total_mots += len(word_tokenize(paragraph['context']))
            nb_words_phrase_final += len(word_tokenize(sent_tokenize(paragraph['context'])[-1]))

print("nombre de paragraphe", nb_paragraphe)
print("nombre de phrase total", nb_phrase_total)
print("nombre moyen de phrase par phrase", nb_total_mots/float(nb_phrase_total))
print("nombre moyen de mots par paragraphe", nb_total_mots/float(nb_paragraphe))
print("nombre moyen de phrase par paragraphe", nb_phrase_total/float(nb_paragraphe))
print("nombre moyen de phrase par paragraphe", nb_phrase_total/float(nb_paragraphe))
print("nombre moyen de mots dans la derniere phrase", nb_words_phrase_final/float(nb_paragraphe))
print((nb_words_phrase_final/float(nb_paragraphe)) / (nb_total_mots/float(nb_paragraphe)))