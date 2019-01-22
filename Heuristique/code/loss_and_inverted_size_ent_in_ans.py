'''
Mesure de la perte et de la taille inversée avec la detection des spans réponse par entité nommée
Ici on va compter le nombre de fois que la reponse se trouve dans une des entiutées trouvées sur le paragraphe
'''

import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *

import json
import sys
import time
import spacy


path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_results/'
selected_data = "dev"
# selected_data = "train"

# type_question = 1 #'Where?'
# type_question = 2 #'How much / many?'
# type_question = 3 #'What name / is called?'
# type_question = 4 #'Who?'
# type_question = 5 #'When / What year?'

type_question = 2

print("type de la question", type_question)
print()


def interesting_entities(type_question):
    if type_question == 'Where?':
        interisting_entities = ("GPE", "LOC", "FAC", "ORG")
    elif type_question == 'How much / many?' :
        interisting_entities = ("MONEY","QUANTITY","PERCENT", "CARDINAL", "TIME","DATE", "ORDINAL")
    elif type_question == 'What name / is called?':
        interisting_entities = ("PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE",'FAC')
    elif type_question == 'Who?':
        interisting_entities = ("PERSON","ORG","NORP","GPE","PRODUCT")
    elif type_question == 'When / What year?':
        interisting_entities = ("TIME","DATE","EVENT")
    return interisting_entities


time1 = time.time()
nlp = spacy.load('en_core_web_sm')





list_type_question_interesting = ['Where?', 'How much / many?', 'What name / is called?', 'Who?', 'When / What year?']
dict_type_question= {}
for type_question in list_type_question_interesting :
    dict_type_question[type_question] = [0, 0, 0] # nb de question observees, perte , taille inverse
num_quest = 0
with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()

    for data in d['data']:
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                type_question = ClassifyQuestion(question['question'])
                if not type_question in list_type_question_interesting:
                    continue
                num_quest += 1
                if num_quest % 100 == 0:
                    # print(dict_type_question)
                    moy_loss_all_types = moy_inverted_size_all_types = total_nb_questions = 0


                    print(num_quest)
                    # time.sleep(2)
                    
                    
                nlp_paragraph = nlp(paragraph['context'])
                list_ent_data = []
                for ent in nlp_paragraph.ents:
                    if ent.label_ in interesting_entities(type_question):
                        list_ent_data.append(ent.text)
                
                dict_type_question[type_question][0] += 1
                concatenation_ent = ' '.join(list_ent_data)
                for answer in list(set(list(map(lambda x: normalize_answer(x["text"]),question['answers'])))):
                    if answer in normalize_answer(concatenation_ent) :
                        dict_type_question[type_question][1] += 1
                        break
                dict_type_question[type_question][2] += len(concatenation_ent) / float(len(paragraph['context']))

for key in dict_type_question :
    moy_loss = 1.0 - (dict_type_question[key][1] / dict_type_question[key][0])
    moy_inverted_size = dict_type_question[key][2] / dict_type_question[key][0]
    moy_loss_all_types += moy_loss * dict_type_question[key][0]
    moy_inverted_size_all_types += moy_inverted_size * dict_type_question[key][0]
    total_nb_questions += dict_type_question[key][0]
    print(key , "   nb de questions",  dict_type_question[key][0], "    perte:", moy_loss , "    taille_inversee moyenne", moy_inverted_size )
print("MOYENNE SUR TOUT LES TYPES",  "   nb de questions",  total_nb_questions, "    perte:", moy_loss_all_types/total_nb_questions , "    taille_inversee moyenne", moy_inverted_size_all_types/total_nb_questions )
print("temps execution", time.time()-time1)
