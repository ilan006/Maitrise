from utils import *
from flair.models import SequenceTagger
from flair.data import Sentence
tagger = SequenceTagger.load('ner-ontonotes')
tagger_pos = SequenceTagger.load('pos')
import corenlp.client as c
client = c.CoreNLPClient(annotators="tokenize ssplit pos lemma parse".split())

# def reduce_Question(question):
#     '''
#     fonction qui va reduire la fonction au éléments les plus important l'aide de PoS
#     :param question: la question (string)
#     :return: les mots important de la question
#     '''
#     sentence = Sentence(normalize_answer(question))
#     # predict NER tags
#     tagger_pos.predict(sentence)
#     rest_of_question = ''
#     for entity in sentence.get_spans('pos'):
#         if entity.tag in ['.', 'WRB', 'WP', 'WDT', 'WP$']:
#             continue
#         rest_of_question += entity.text + ' '
#     return rest_of_question
#
#
# def reduce_Question(question):  ##Reduce1
#     '''
#     fonction qui va reduire la fonction au éléments les plus important l'aide de PoS
#     :param question: la question (string)
#     :return: les mots important de la question
#     '''
#
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)
#
#     def white_space_fix(text):
#         return ' '.join(text.split())
#
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)
#
#     return white_space_fix(remove_articles(remove_punc(question)))



def reduce_Question(question_str):
    question = client.annotate(question_str)
    tokens = question.sentence[0].token
    wh_token_index = None
    for i, token in enumerate(tokens):
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            wh_token_index = i
            break
    if wh_token_index is None: #Si il n'y a pas de mot interogatiphe on utilisera l'ensemble de la question.
        return question_str

    mots_conserves = [tokens[wh_token_index].word]
    for edge in question.sentence[0].basicDependencies.edge:
        if edge.dep in ['punct', 'cop', 'root']:
            continue

        if edge.source == wh_token_index + 1:
            if tokens[edge.target - 1].lemma == 'in':
                continue
            mots_conserves.append(tokens[edge.target - 1].word)

        if edge.target == wh_token_index + 1:
            mots_conserves.append(tokens[edge.source - 1].word)
    print(mots_conserves)
    return ' '.join(mots_conserves)