import json
import sys
path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_txt/'

sys.path.append('../..')
sys.path.append('../../utils/')
from utils.get_numeral_sentence import num_sentence
from gensim.models import Word2Vec
from nltk import sent_tokenize
from nltk import sent_tokenize,word_tokenize
# from utils.get_numeral_sentence import numeral_sentence
# from utils.compair_with_embeding import get_best_sentence
# import re
# import string
# model = Word2Vec.load(path_dest + 'embeding.txt')

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
with open(path_dest + 'data.txt', 'w') as outfile:
    num_quest = 0
    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                list_ans = []
                outfile.write("question " + str(question['id']) + " = " + question['question'] + "\n")
                for answer in question['answers']:
                    if (num_quest % 1000) == 0 : print(num_quest)
                    pos = answer['answer_start']
                    sentence_position = num_sentence(pos, paragraph['context'])
                    if not (sentence_position in list_ans):
                        list_ans.append(sentence_position)
                        outfile.write("phrase " + str(question['id']) + " = " + sent_tokenize(paragraph['context'])[sentence_position]+ "\n")
                        outfile.write("réponse " + str(question['id']) + " = " + answer["text"] + "\n")
                        num_quest += 1
                # list_ans = []
                # list_ans = [get_best_sentence(model, map(normalize_answer,sent_tokenize(paragraph['context'])),normalize_answer(question['question']).split())]
                # # for answer in question['answers']:
                # #     pos = answer['answer_start']
                # #     sentence_position = numeral_sentence(pos, paragraph['context'], False)
                # #     if not(sentence_position in list_ans) :
                # #         list_ans.append(sentence_position)
                # # list_ans = [21.0]
                # out_json[question['id']] = list_ans #test

#
# with open(path_dest+'data_toTest.json', 'w') as outfile:
#     json.dump(out_json, outfile)