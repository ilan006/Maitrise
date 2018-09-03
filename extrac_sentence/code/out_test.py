import json
import sys
path_data = '../../data/'
path_dest = '../../data_perso/'

sys.path.append("../../")

from gensim.models import Word2Vec
from nltk import sent_tokenize
from utils.get_numeral_sentence import numeral_sentence
from utils.compair_with_embeding import get_best_sentence
import re
import string
model = Word2Vec.load(path_dest + 'embeding.txt')

out_json = {}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    num_quest = 1
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                if (num_quest % 1000) == 0 : print(num_quest)
                num_quest += 1
                list_ans = []
                list_ans = [get_best_sentence(model, map(normalize_answer,sent_tokenize(paragraph['context'])),normalize_answer(question['question']).split())]
                # for answer in question['answers']:
                #     pos = answer['answer_start']
                #     sentence_position = numeral_sentence(pos, paragraph['context'], False)
                #     if not(sentence_position in list_ans) :
                #         list_ans.append(sentence_position)
                # list_ans = [21.0]
                out_json[question['id']] = list_ans


with open(path_dest+'data_toTest.json', 'w') as outfile:
    json.dump(out_json, outfile)