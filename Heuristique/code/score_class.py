'''
Classe qui va comptabiliser les score pour un modèle
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *
import numpy as np




class Score:
    list_keys = ['frequence', 'exact_match','f1','pred_in_ans', 'loss', 'one_of_pred_in_one_of_ans', 'inverted_size']

    def __init__(self):
        self.dict_resultats = {} #dictionnaire contenant les resultats pour chacune des questions

    def __str__(self):
        dict_str_final = "Question ; " + " ; ".join(Score.list_keys) + "\n"
        for type_question in self.dict_resultats:
            list_elems_item = list(map(lambda key : str(self.dict_resultats[type_question][key]) , Score.list_keys))
            dict_str_final += type_question + " ; " + " ; ".join(list_elems_item) + "\n"

        list_elems_avg = list(map(lambda x: str(x) , self.get_average_score()))
        dict_str_final += "MOYENNE TOUT TYPE ; "+" ; ".join(list_elems_avg) + "\n"
        return dict_str_final

    def add_score(self, type_question: str, prediction, list_predictions, list_answers, len_paragraph):

        try:
            self.dict_resultats[type_question]['frequence'] += 1
        except:
            self.dict_resultats[type_question] = {}
            self.dict_resultats[type_question]['frequence'] = 1
            self.dict_resultats[type_question]['exact_match'] = 0 #la prediction est exactement égale a une réponse
            self.dict_resultats[type_question]['f1'] = 0    #score f1 entre la prédiction et les réponse
            self.dict_resultats[type_question]['pred_in_ans'] = 0 #la prédiction est dans la réponse
            self.dict_resultats[type_question]['loss'] = 0 #aucune des réponse n'est inclus dans aucun élément de l'ensemble de prédictions
            self.dict_resultats[type_question]['one_of_pred_in_one_of_ans'] = 0 #une des prédiction se trouve dans une des réponse
            self.dict_resultats[type_question]['inverted_size'] = 0 # le ratio : taille de l'ensemble de prédictions / la taille du parragraphe

        if len(list_predictions) == 0:
            return

        ground_truths = list(map(lambda x: normalize_answer(x['text']), list_answers))
        list_prediction_in_ans = list(map(lambda x: prediction in x, ground_truths))
        list_ans_in_predictions = list(map(lambda x: max(list(map(lambda y: x in y, list_predictions))), ground_truths))
        list_predictions_in_ans = list(map(lambda x: max(list(map(lambda y: x in y, ground_truths))), list_predictions))
        concatenation_predictions = ' '.join(list_predictions)

        self.dict_resultats[type_question]['exact_match'] += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        self.dict_resultats[type_question]['f1'] += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        self.dict_resultats[type_question]['pred_in_ans'] += max(list_prediction_in_ans)
        self.dict_resultats[type_question]['loss'] += max(list_ans_in_predictions)
        self.dict_resultats[type_question]['one_of_pred_in_one_of_ans'] += max(list_predictions_in_ans)
        self.dict_resultats[type_question]['inverted_size'] += len(concatenation_predictions) / float(len_paragraph)

    def get_average_score(self):
        tab_averages = [0] * 7
        for type_question in self.dict_resultats:
            freq = self.dict_resultats[type_question]['frequence']
            list_results = [freq] + list(map(lambda key: self.dict_resultats[type_question][key] * freq, Score.list_keys[1:]))
            tab_averages = [x + y for x, y in zip(tab_averages, list_results)]

        tab_averages = [tab_averages[0]] +  [x / tab_averages[0] for x in tab_averages]
        return tab_averages