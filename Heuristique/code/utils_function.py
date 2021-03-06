'''
Classe avec toute les fonctions utilisé pour retourner le span réponse
'''
import spacy
nlp_spacy = spacy.load('en_core_web_sm')
nlp_spacy.entity.set_annotations

from interesting_entities import *
from segtok.segmenter import split_single

from flair.models import SequenceTagger
from flair.data import Sentence
tagger = SequenceTagger.load('ner-ontonotes')
tagger_pos = SequenceTagger.load('pos')

from nltk import ngrams , word_tokenize

from utils import *

from random import randint
# import fastText
# model_fastText = fastText.load_model('../../../Divers_Data_Maitrise/wiki.simple/wiki.simple.bin')

class Function_prediction:
#
    available_methods = ['first', 'embeding', 'random','Qembeding']
    available_models = ['ner_PRG_spacy', 'ner_SENT_spacy','ner_PRG_flair', 'ner_SENT_flair','all_ner_PRG_spacy', 'ngram' ]
    list_type_question_interesting = ['Where?', 'How much / many?', 'What name / is called?', 'Who?', 'When / What year?']

    def __init__(self, type_method: str, model : str ,param_ngram = 1 ):
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
        self._interesting_entities_method = None
        self.param_ngram = param_ngram
        self.model = model
        self.assign_funct_predict()
        self.assign_model_predict()

    def assign_funct_predict(self):
        '''
        Function qui va assigner la methode utiliser et sa description
        '''
        if self.type_method == 'first':
            self._predict_function = Function_prediction.get_first
            self._description = 'On retourne la prédiction  apparue le plus tot dans le text.'
        elif self.type_method == 'random':
            self._predict_function = Function_prediction.get_random
            self._description = 'On retourne une des prédictions de manière aléatoire.'
        elif self.type_method == 'embeding':
            self._predict_function = Function_prediction.get_embeding
            self._description = 'On retourne la prediction la plus proche de la question (embeding cosine).'
        self._description += 'On s\'interesse ici juste à la phrase contenant la première réponse.'

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

    # def get_function(self):
    #     '''
    #     :return: la function utilisée
    #     '''
    #     return self._predict_function

############## Determination du rang séléctionner
    def get_first(*args):
        '''
        fonction qui va retourner 0 (le premier élément d'une liste)
        '''
        return 0


    def get_random(list_predictions, *args):
        '''
        fonction qui va retourner aléatoirement une des prediction
        @:param list_predictions :liste des prediction possible
        '''
        return randint(0,len(list_predictions)-1)

    def get_embeding(list_predictions, question_str):
        '''
        fonction qui va retourner la prédiction la plus proche en fonction des embeding
        @:param list_predictions :liste des prediction possible
        '''
        question_embeding = get_fastText_embeding(question_str)
        list_embedings_ent = list(map(lambda x: get_fastText_embeding(x), list_predictions))

        # question_embeding = avg_sentence_vector(question_str, model_fastText, with_steming=False)
        # list_embedings_ent = list(map(lambda x: avg_sentence_vector(x, model_fastText, with_steming = False), list_predictions))
        list_cosine_embeding_question_ent = list(map(lambda x: abs(cosine_similarity(x,question_embeding )), list_embedings_ent))
        rank_of_prediction = np.argmax(list_cosine_embeding_question_ent)
        return rank_of_prediction
################


    def predict(self, list_predictions, question_str, bool_normalise=True):
        list_predictions_copy = list_predictions.copy()
        question_str_copy = question_str
        if not list_predictions_copy:
            return ''
        elif bool_normalise:
            list_predictions_copy = list(map(lambda elem : normalize_answer(elem), list_predictions_copy))
            question_str_copy = normalize_answer(question_str_copy)
        rank_predict = self._predict_function(list_predictions_copy, question_str_copy)
        return list_predictions[rank_predict]

    def interesting_questions(type_question):
        return type_question in Function_prediction.list_type_question_interesting

    def all_questions(type_question):
        '''
        return toujours vrai
        '''
        return True

    def interesting_question_test(self, type_question):
        '''
        function pour savoir si une question est prise en compte dans le modèl
        :param type_question: le type de la question
        :return: un Boolean
        '''
        return self._interesting_entities_method(type_question)

    def assign_model_predict(self):
        '''
        Function qui va assigner un model de prediction
        '''
        if self.model == 'ner_PRG_spacy':
            self._list_predictions_function = Function_prediction.model_ner_PRG_spacy
            self._interesting_entities_method = Function_prediction.interesting_questions
            self.model_description = 'Prédictions realisées sur l\'ensemble du text avec spacy.'
        elif self.model == 'ner_SENT_spacy':
            self._list_predictions_function = Function_prediction.model_ner_SENT_spacy
            self._interesting_entities_method = Function_prediction.interesting_questions
            self.model_description = 'Prédictions realisées sur chacune des phrases du text avec spacy.'
        elif self.model == 'ner_PRG_flair':
            self._list_predictions_function = Function_prediction.model_ner_PRG_flair
            self._interesting_entities_method = Function_prediction.interesting_questions
            self.model_description = 'Prédictions realisées sur l\'ensemble du text avec flair.'
        elif self.model == 'ner_SENT_flair':
            self._list_predictions_function = Function_prediction.model_ner_SENT_flair
            self._interesting_entities_method = Function_prediction.interesting_questions
            self.model_description = 'Prédictions realisées sur chacune des phrases du text avec flair.'
        elif self.model == 'NP_PRG_spacy':
            self._list_predictions_function = Function_prediction.model_NP_PRG_spacy
            self._interesting_entities_method = Function_prediction.all_questions
            self.model_description = 'Prédictions realisées sur l\'ensemble du text avec spacy.'
        elif self.model == 'NP_ner_PRG_spacy':
            self._list_predictions_function = Function_prediction.model_NP_ner_PRG_spacy
            self._interesting_entities_method = Function_prediction.all_questions
            self.model_description = 'Prédictions realisées sur l\'ensemble du text avec spacy.'
        elif self.model == 'all_ner_PRG_spacy':
            self._list_predictions_function = Function_prediction.model_all_ner_PRG_spacy
            self._interesting_entities_method = Function_prediction.all_questions
            self.model_description = 'Prédictions realisées sur l\'ensemble du text avec spacy.'
        elif self.model == 'ngram':
            self._list_predictions_function = self.n_gram
            self._interesting_entities_method = Function_prediction.all_questions
            self.model = str(self.param_ngram)+'-gram'
            self.model_description = 'On a recupéré les ' + self.model + ' ou inferieur sur l\'ensemble du text'


    def get_list_predictions(self,paragraph , type_question = None):
        return self._list_predictions_function(paragraph,type_question)

    def n_gram(self,text,*args):
        '''
        fonction qui va retourner l'ensemble des ngrams inferieur au paramètre entrer
        :return:
        '''
        text_tokenize = word_tokenize(text)
        list_ngrams = []
        ngrams_all = list(map(lambda n: ngrams(text_tokenize, n), range(1, self.param_ngram+1)))
        for grams in ngrams_all:
            list_ngrams += list(set(map(lambda x: ' '.join(x), grams)))#####################retire normalisation
        list_ngrams = list(filter(None, list_ngrams))
        return list_ngrams


    def model_all_ner_PRG_spacy(paragraph , type_question):
        '''
        fonciton qui va retourner l'ensemble des entité nommé detecter dans le text
        :param type_question:
        :return:
        '''
        nlp_paragraph = nlp_spacy(paragraph)
        list_predictions_data = []
        for entity in nlp_paragraph.ents:
            if entity.label_ in interesting_entities('all') and len(normalize_answer(entity.text)):
                list_predictions_data.append(entity.text)
        return list_predictions_data

    def model_ner_PRG_spacy(paragraph , type_question):
        nlp_paragraph = nlp_spacy(paragraph)
        list_predictions_data = []
        for entity in nlp_paragraph.ents:
            if entity.label_ in interesting_entities(type_question) and len(normalize_answer(entity.text)):
                list_predictions_data.append(entity.text)
        return list_predictions_data

    def model_ner_SENT_spacy(paragraph , type_question):
        sentences = [nlp_spacy(sent) for sent in split_single(paragraph)]
        list_predictions_data = []
        for sentence in sentences:
            for entity in sentence.ents:
                if entity.label_ in interesting_entities(type_question) and len(normalize_answer(entity.text)):
                    list_predictions_data.append(entity.text)
        return list_predictions_data

    def model_ner_PRG_flair(paragraph , type_question):
        sentence = Sentence(paragraph)
        tagger.predict(sentence)
        list_predictions_data = []
        for entity in sentence.get_spans('ner'):
            if entity.tag in interesting_entities(type_question) and len(normalize_answer(entity.text)):
                list_predictions_data.append(entity.text)
        return list_predictions_data

    def model_ner_SENT_flair(paragraph , type_question):
        sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(paragraph)]
        tagger.predict(sentences)
        list_predictions_data = []
        for sentence in sentences:
            for entity in sentence.get_spans('ner'):
                if entity.tag in interesting_entities(type_question) and len(normalize_answer(entity.text)):
                    list_predictions_data.append(entity.text)
        return list_predictions_data

    def model_NP_PRG_spacy(paragraph , *args):
        '''
        Fonction qui va retourner tout les groupe nominaux dans une text
        :return: la liste des groupes nominaux contenus dans le text
        '''
        nlp_paragraph = nlp_spacy(paragraph)
        list_predictions_data = []
        for entity in nlp_paragraph.noun_chunks:
            if len(normalize_answer(entity.text)):
                list_predictions_data.append(normalize_answer(entity.text))
        return list_predictions_data


    def model_NP_ner_PRG_spacy(paragraph, type_question):
        ner_list = Function_prediction.model_ner_PRG_spacy(paragraph, type_question)
        noun_phrase_list = Function_prediction.model_NP_PRG_spacy(paragraph)
        return ner_list + noun_phrase_list


