'''
Programme qui affiche tours à tours les entité (et leurs labels) pour des phrases données
'''

import spacy
nlp = spacy.load('en_core_web_sm')
sentence = "   "
while len(sentence) > 0 :
    sentence = input('phrase?: ')
    nlp_sentence= nlp(sentence)

    for ent in nlp_sentence.ents:
        print(ent.text, ent.label_)