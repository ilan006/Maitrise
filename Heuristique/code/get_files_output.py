'''
fonction qui va cree un ensemble de fichiers contenant l'ensemble des réponses produites
'''
import os
from utils import *

def clear_directory(path):
    '''
    fonction qui va remetre les fichier a 0
    :return:
    '''
    for file in os.listdir(path):
        os.remove(path+file)


def add_trace(path, question, text, set_predictions, final_prediction):
    '''
    fonction qui va ajouter la trace d'un question
    :return:
    '''

    question_str = question['question']
    type_question = ClassifyQuestion(question_str)
    name_file = type_question.replace(" ", "_") + '.txt'
    id = question['id']
    set_answers = set(map(lambda x: x["text"], question['answers']))

    with open(path + name_file, 'a') as outfile:
        outfile.write("question " + str(id) + " = " + question_str + "\n") #la question sous forme d'une string
        outfile.write("text " + str(id) + " = " + text + "\n") # le texte dans lequelle on doit trouver la réponse
        outfile.write("answers " + str(id) + " = " + str(set_answers) + "\n")
        outfile.write("set_predictions" + str(id) + " = " + str(set_predictions) + "\n")
        outfile.write("final_prediction" + str(id) + " = " + str(final_prediction) + "\n")
        outfile.write(" \n")

clear_directory('../traces/')