import json
import sys
path_data = '../../../Data_Maitrise/data/'
path_dest = '../../../Data_Maitrise/data_perso/'

sys.path.append("../../")
from nltk import sent_tokenize
from nltk import word_tokenize

dict = {}
nb_answers = 0
total_words = 0
with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                list_ans = []
                for answer in question['answers']:
                    span = answer['text']
                    if span not in list_ans:
                        list_ans.append(span)
                        # nb_words_ans = len(span.split())
                        nb_words_ans = len(word_tokenize(span))
                        total_words += nb_words_ans
                        nb_answers += 1
                        if nb_words_ans in dict:
                            dict[nb_words_ans] += 1
                        else:
                            dict[nb_words_ans] = 1
                    if nb_answers % 1000 ==0 : print(nb_answers)
print(nb_answers)
print(float(total_words)/nb_answers)


# on écrit la distribution des spans
with open('../taille_span_dev.csv', 'w') as f:
    item_str = "taille" + "," + "compte relatif" + "," + "compte" + "\n"
    f.write(item_str)
    for key in dict:
        # item_str = str(key) + "," + str((dict[key]/ float(nb_answers))) +"," +str(dict[key]) + "\n"
        item_str = str(key) + "," + str(round(dict[key] / float(nb_answers),4)) + "," + str(dict[key]) + "\n"
        f.write(item_str)