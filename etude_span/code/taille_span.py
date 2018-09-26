'''
programme qui retourne dans un fichier la distribution des tailles des span-reponses
'''

import json
path_data = '../../../Data_Maitrise/data/'
path_dest = '../graph/'

from nltk import word_tokenize

dict = {}
nb_answers = 0
total_words_ans = 0


# total_deux_span = 0
with open(path_data+'dev-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:

                # bool_deja_fait = False


                list_ans = []
                for answer in question['answers']:
                    span = answer['text']
                    if span not in list_ans:
                        list_ans.append(span)
                        nb_words_ans = len(word_tokenize(span))
                        total_words_ans += nb_words_ans
                        nb_answers += 1
                        if nb_words_ans in dict:
                            dict[nb_words_ans] += 1
                        else:
                            dict[nb_words_ans] = 1
                        # if len(word_tokenize(span)) == 2 and not(bool_deja_fait):
                        #     total_deux_span += 1
                        bool_deja_fait = True
                    if nb_answers % 1000 == 0:
                        print(nb_answers)

print('le nombre de span differents:', nb_answers)
print('la taille moyenne des spans réponse:', float(total_words_ans)/nb_answers)
# print(total_deux_span)

# on écrit la distribution des spans
with open(path_dest + 'taille_span_dev.csv', 'w') as f:
    item_str = "taille" + "," + "compte relatif" + "," + "compte" + "\n"
    f.write(item_str)
    for key in dict:
        # item_str = str(key) + "," + str((dict[key]/ float(nb_answers))) +"," +str(dict[key]) + "\n"
        item_str = str(key) + "," + str(round(dict[key] / float(nb_answers),4)) + "," + str(dict[key]) + "\n"
        f.write(item_str)