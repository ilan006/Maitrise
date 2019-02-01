'''
Classe avec toute les fonctions utilisé pour retourner le span réponse
'''
import spacy
nlp_spacy = spacy.load('en_core_web_sm')

from interesting_entities import *
from segtok.segmenter import split_single

from flair.models import SequenceTagger
from flair.data import Sentence
tagger = SequenceTagger.load('ner-ontonotes')

from utils import *

class Function_prediction:
#
    available_methods = ['first', 'embeding', 'random']
    available_models = ['PRG_spacy', 'SENT_spacy','PRG_flair', 'SENT_flair' ]
    def __init__(self, type_method: str, model : str):
        try:
            type_method in Function_prediction.available_methods
            model in Function_prediction.available_models
        except:
            print('Le type de méthode demandé n\'existe pas')
        self.type_method = type_method
        self._description = None
        self._predict_function = None
        self._list_predictions_function = None
        self.model_description = None
        self.model = model
        self.assign_funct_predict()
        self.assign_model_predict()



    def get_first(list_predictions):
        '''
        fonction qui va retourner le premier élément de la liste de prédiction
        @:param list_predictions :liste des prediction possible
        '''
        if len(list_predictions) == 0:
            return ''
        else:
            return list_predictions[0]
    def assign_funct_predict(self):
        '''
        Function qui va assigner la methode utiliser et sa description
        '''
        if self.type_method == 'first':
            self._predict_function = Function_prediction.get_first
            self._description = 'On retourne la prédiction  apparue le plus tot dans le text.'

    def get_description(self):
        '''
        :return: la description textuelle de la methode utilisée
        '''
        return self._description



    def get_type_method(self):
        '''
        :return: le type textuelle de la methode utilisée
        '''
        return self.type_method

    def get_function(self):
        '''
        :return: la function utilisée
        '''
        return self._predict_function

    def predict(self,list_predictions):
        return self._predict_function(list_predictions)



    def assign_model_predict(self):
        '''
        Function qui va assigner un model de prediction
        '''
        if self.model == 'PRG_spacy':
            self._list_predictions_function = Function_prediction.model_PRG_spacy
            self.model_description = 'Prédictions realisées sur l\'ensemble du paragraphe avec spacy'
        elif self.model == 'SENT_spacy':
            self._list_predictions_function = Function_prediction.model_SENT_spacy
            self.model_description = 'Prédictions realisées sur chacune des phrases avec spacy'
        elif self.model == 'PRG_flair':
            self._list_predictions_function = Function_prediction.model_PRG_flair
            self.model_description = 'Prédictions realisées sur l\'ensemble du paragraphe avec flair'
        elif self.model == 'SENT_flair':
            self._list_predictions_function = Function_prediction.model_SENT_flair
            self.model_description = 'Prédictions realisées sur chacune des phrases avec flair'

    def get_list_predictions(self,paragraph , type_question = None):
        return self._list_predictions_function(paragraph,type_question)

    def model_PRG_spacy(paragraph , type_question):
        nlp_paragraph = nlp_spacy(paragraph)
        list_predictions_data = []
        for ent in nlp_paragraph.ents:
            if ent.label_ in interesting_entities(type_question):
                list_predictions_data.append(normalize_answer(ent.text))
        return list_predictions_data

    def model_SENT_spacy(paragraph , type_question):
        sentences = [nlp_spacy(sent) for sent in split_single(paragraph)]
        list_predictions_data = []
        for sentence in sentences:
            for ent in sentence.ents:
                if ent.label_ in interesting_entities(type_question):
                    list_predictions_data.append(normalize_answer(ent.text))
        return list_predictions_data

    def model_PRG_flair(paragraph , type_question):
        sentence = Sentence(paragraph)
        tagger.predict(sentence)
        list_predictions_data = []
        for entity in sentence.get_spans('ner'):
            if entity.tag in interesting_entities(type_question):
                list_predictions_data.append(normalize_answer(entity.text))
        return list_predictions_data

    def model_SENT_flair(paragraph , type_question):
        sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(paragraph)]
        tagger.predict(sentences)
        list_predictions_data = []
        for sentence in sentences:
            for entity in sentence.get_spans('ner'):
                if entity.tag in interesting_entities(type_question):
                    list_predictions_data.append(normalize_answer(entity.text))
        return list_predictions_data