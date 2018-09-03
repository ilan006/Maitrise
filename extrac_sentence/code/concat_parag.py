import json
import sys
from nltk import sent_tokenize
import pickle

path_data = '../../data/'
path_dest = '../../data_perso/'

with open(path_data + 'train-v1.1.json') as json_data:
    d = json.load(json_data)
    json_data.close()
    with open(path_dest + 'train_concat.txt', 'w') as f:
        for data in d['data']:
            for paragraph in data['paragraphs']:
                f.write(paragraph['context'])
