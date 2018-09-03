# coding= utf-8


import pickle
import csv
import os
import operator

dict = pickle.load(open("../dict/freq_sentence_positionRelative.pickle", 'rb'))

csv_file = '../dict/mycsvfile.csv'

dict = sorted(dict.items(), reverse=False, key=operator.itemgetter(0))
with open('../dict/mycsvfile.csv', 'w') as f:
    for item in dict:
        item_str = (str(item[0]),(str(item[1])))
        item_str = " ; ".join(item_str) + "\n"
        f.write(item_str)

