"""
Dans ce programme on va realiser la matrice de similarité entre la phrase contenant le span réponse et la question.
"""
import json
import time
path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/Matrix_similarite/'

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

from utils.get_numeral_sentence import num_sentence
from nltk import sent_tokenize,word_tokenize
from nltk import sent_tokenize
import pandas as pd

#ignorer toutes les depreciations de fonctions
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
out_json = {}


import fastText
model = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')




def affiche_table_cosine(question, sequence):
    list_words_question = word_tokenize(question)
    list_words_sequence = word_tokenize(sequence)
    matrix_alignement = np.zeros((len(list_words_sequence),len(list_words_question)))

    for i, word_sentence in enumerate(list_words_sequence):
        vect_word_sentence = model.get_word_vector(word_sentence)
        for j, word_question in enumerate(list_words_question):
            vect_word_question = model.get_word_vector(word_question)
            sim = cosine_similarity(vect_word_sentence, vect_word_question)
            matrix_alignement[i][j] = sim
    return matrix_alignement


with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()
# with open(path_dest + 'data.txt', 'w') as outfile:
    num_quest = 0
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                list_ans = []
                # outfile.write("question " + str(question['id']) + " = " + question['question'] + "\n")
                for answer in question['answers']:
                    if (num_quest % 100) == 0 : print(num_quest)
                    pos = answer['answer_start']
                    sentence_position = num_sentence(pos, paragraph['context'], answer['text'])
                    if not (sentence_position in list_ans):
                        phrase_str = (sent_tokenize(paragraph['context'])[sentence_position]).lower()
                        question_str = str(question['question']).lower()
                        list_ans.append(sentence_position)
                        A = affiche_table_cosine(question_str,phrase_str)
                        df = pd.DataFrame(A, index= word_tokenize(phrase_str), columns=word_tokenize(question_str))
                        with open('test/df' + str(question['id']) + '.csv', 'a') as out_file:
                            df.to_csv(out_file, index=True, header=True, sep=',')
                        num_quest += 1


