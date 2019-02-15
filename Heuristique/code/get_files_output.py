'''
fonction qui va cree un ensemble de fichiers contenant l'ensemble des réponses produites
'''
import os
from utils import *
from score_class import *

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
    # name_file = type_question.replace(" ", "_") + '.txt'
    name_file = type_question.replace("/", "|") + '.txt'
    id = question['id']
    set_answers = set(map(lambda x: x["text"], question['answers']))

    with open(path + name_file, 'a') as outfile:
        if(get_loss(set_answers,set_predictions)) :
            outfile.write("ERROR " + str(id))
        outfile.write("QUESTION " + str(id) + " = " + question_str + "\n   \n") #la question sous forme d'une string
        outfile.write("TEXT " + str(id) + " = " + text + "\n   \n") #le texte dans lequelle on doit trouver la réponse
        outfile.write("ANSWERS " + str(id) + " = " + str(set_answers) + "\n   \n")
        outfile.write("SET_PREDICTIONS " + str(id) + " = " + str(set_predictions) + "\n   \n")
        outfile.write("FINAL_PREDICTION " + str(id) + " = " + str(final_prediction) + "\n")
        outfile.write(" \n  \n------------------  \n   \n  \n")

clear_directory('../traces/')