'''
Classe qui va comptabiliser les score pour un modèle
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *
import numpy as np
import time



class Score:
    list_keys = ['frequence', 'exact_match','f1','pred_in_ans', 'loss', 'one_of_pred_in_one_of_ans', 'inverted_size', 'nb_preds', 'nb_preds_differentes']

    def __init__(self):
        self.dict_resultats = {} #dictionnaire contenant les resultats pour chacune des questions

    def __str__(self):
        dict_str_final = "Question ; " + " ; ".join(Score.list_keys) + "\n \n"
        for type_question in self.dict_resultats:
            # list_elems_item = list(map(lambda key : str(self.dict_resultats[type_question][key]) , Score.list_keys))
            list_elems_item = list(map(lambda x: str(x) , self.get_average_score_for_type(type_question)))
            dict_str_final += type_question + " ; " + " ; ".join(list_elems_item) + "\n"

        list_elems_avg = list(map(lambda x: str(x) , self.get_average_score_total()))
        dict_str_final += "MOYENNE TOUT TYPE ; "+" ; ".join(list_elems_avg) + "\n"
        return dict_str_final

    def add_score(self, type_question: str, prediction, list_predictions, list_answers, len_paragraph):
        try:
            self.dict_resultats[type_question]['frequence'] += 1
        except:
            self.dict_resultats[type_question] = {}
            self.dict_resultats[type_question]['frequence'] = 1.0
            self.dict_resultats[type_question]['exact_match'] = 0.0 #la prediction est exactement égale a une réponse
            self.dict_resultats[type_question]['f1'] = 0.0    #score f1 entre la prédiction et les réponse
            self.dict_resultats[type_question]['pred_in_ans'] = 0.0 #la prédiction est dans la réponse
            self.dict_resultats[type_question]['loss'] = 0.0 #aucune des réponse n'est inclus dans aucun élément de l'ensemble de prédictions
            self.dict_resultats[type_question]['one_of_pred_in_one_of_ans'] = 0.0 #une des prédiction se trouve dans une des réponse
            self.dict_resultats[type_question]['inverted_size'] = 0.0 # le ratio : taille de l'ensemble de prédictions / la taille du parragraphe
            self.dict_resultats[type_question]['nb_preds'] = 0.0  # le nombre total de prédiction
            self.dict_resultats[type_question]['nb_preds_differentes'] = 0.0  # le nombre total de prédiction differentes

        ground_truths = list(map(lambda x:normalize_answer( x['text']), list_answers))
        list_prediction_in_ans = list(map(lambda x: normalize_answer(prediction) in x, ground_truths))


        try:
            list_ans_in_predictions = list(map(lambda x: max(list(map(lambda y: x in y, list_predictions))), ground_truths))
            list_predictions_in_ans = list(map(lambda x: max(list(map(lambda y: x in y, ground_truths))), list_predictions))
        except:
            list_ans_in_predictions = [0]
            list_predictions_in_ans = [0]
        concatenation_predictions = ' '.join(list_predictions)
        #
        # if len(list_predictions) == 0:
        #     return

        self.dict_resultats[type_question]['exact_match'] += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        self.dict_resultats[type_question]['f1'] += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        self.dict_resultats[type_question]['pred_in_ans'] += max(list_prediction_in_ans)
        self.dict_resultats[type_question]['loss'] += get_loss(ground_truths, list_predictions)
        self.dict_resultats[type_question]['one_of_pred_in_one_of_ans'] += max(list_predictions_in_ans)
        self.dict_resultats[type_question]['inverted_size'] += len(concatenation_predictions) / float(len_paragraph)
        self.dict_resultats[type_question]['nb_preds'] += len(list_predictions)
        self.dict_resultats[type_question]['nb_preds_differentes'] += len(set(list_predictions))

    def get_average_score_total(self):
        tab_averages = [0] * len(Score.list_keys)
        for type_question in self.dict_resultats:
            list_results = self.get_average_score_for_type(type_question)
            freq = list_results[0]
            list_results = [freq] + list(map(lambda elem: elem * freq, list_results[1:]))
            tab_averages = [x + y for x, y in zip(tab_averages, list_results)]

        tab_averages = [tab_averages[0]] + [x / tab_averages[0] for x in tab_averages[1:]]
        return tab_averages

    def get_average_score_for_type(self, type_question):
        freq = self.dict_resultats[type_question]['frequence']
        # list_results = [freq] + list(map(lambda key: 100.0 * self.dict_resultats[type_question][key] / freq, Score.list_keys[1:7])) + list(map(lambda key: self.dict_resultats[type_question][key] / freq, Score.list_keys[7:]))
        list_results = [freq] + list(map(lambda key: 100.0 * self.dict_resultats[type_question][key] / freq, Score.list_keys[1:7])) + list(map(lambda key: self.dict_resultats[type_question][key] / freq, Score.list_keys[7:]))
        return list_results

def get_loss(list_answers, list_predictions):
    '''
    return true si il est impossible de retrouver la réponse dans l'ensemble de predictions
    :param self:
    :return:
    '''
    if not len(list_predictions):
        return 1
    ground_truths = list(map(lambda x: normalize_answer(x), list_answers))
    list_ans_in_predictions = list(map(lambda x: max(list(map(lambda y: x in y, list_predictions))), ground_truths))
    return 1 - max(list_ans_in_predictions)
