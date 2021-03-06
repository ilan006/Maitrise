'''
On va mesurer les résultats obtenu avec le système d'entitées nommées avec Scipy
'''
import sys
sys.path.append('../..')
sys.path.append('../../utils/')
from utils import *
import tqdm
import json
import time
time1 = time.time()
import os
from score_class import *
from utils_function import *
from get_files_output import *
from Reduce_Question import *
file_name = os.path.basename(__file__)[:-3]

path_data = '../../../Data_Maitrise/data/'
path_dest = '../data_results/'
path_trace = '../traces'
selected_data = "dev"
# selected_data = "train"

# chosen_model = Function_prediction('first', 'ner_PRG_spacy')
# chosen_model = Function_prediction('first', 'ner_SENT_spacy')
# chosen_model = Function_prediction('first', 'ner_PRG_flair')
# chosen_model = Function_prediction('first', 'ner_SENT_flair')
# chosen_model = Function_prediction('random', 'ner_PRG_flair')
# chosen_model = Function_prediction('random', 'ner_PRG_spacy')
# chosen_model = Function_prediction('first', 'NP_PRG_spacy')
# chosen_model = Function_prediction('first', 'NP_ner_PRG_spacy')
# chosen_model = Function_prediction('embeding', 'ner_PRG_spacy')
chosen_model = Function_prediction('embeding', 'all_ner_PRG_spacy')
# param_ngram = 5
# chosen_model = Function_prediction('embeding', 'ngram', param_ngram = param_ngram)
#######################«################################################################################
custom_name =''
custom_name ='_Q_Reduce2'
file_name = file_name + '_' + chosen_model.get_type_method() + '_' + chosen_model.model + custom_name + '.csv'
description_file_str = chosen_model.get_description() +' '+ chosen_model.model_description
########################################################################################################
print(chosen_model.get_type_method() + '_' + chosen_model.model)

path_trace += '_' + chosen_model.get_type_method() + '_' + chosen_model.model + custom_name + '/'

score_model = Score()
clear_directory(path_trace)

num_quest = 0
print("temps d'initialisation", time.time()-time1)
with open(path_data + selected_data + '-v1.1.json', 'r') as input:
    d = json.load(input)
    input.close()
    for data in tqdm.tqdm(d['data']):
        for paragraph in data['paragraphs']:
            for question in paragraph['qas']:
                question_str = question['question']
                type_question = ClassifyQuestion(question_str)
                if not chosen_model.interesting_question_test(type_question):
                    continue
                num_quest += 1
                if num_quest % 100 == 0:
                    with open(path_dest + file_name, 'w') as f:
                        f.write(score_model.__str__())
                    # print(num_quest)
                #     print("temps execution", time.time() - time1)

                question_str = reduce_Question(question_str)

                sentence = get_sentence(question['answers'][0], paragraph['context'])

                ######paragraphe ou phrase
                # text_to_evaluate = paragraph['context']
                text_to_evaluate = sentence
                list_predictions_data = chosen_model.get_list_predictions(text_to_evaluate, type_question)
                prediction = chosen_model.predict(list_predictions_data, question_str, bool_normalise=False)

                score_model.add_score(type_question, prediction, list_predictions_data,  question['answers'], float(len(text_to_evaluate)))
                # score_model.add_score(type_question, prediction, list_predictions_data,  reduce_question, float(len(paragraph['context'])))

                #on sauvegarde la trace de la question dans un fichier
                add_trace(path_trace, question_str, question, text_to_evaluate, list_predictions_data, prediction)

with open(path_dest + 'description.txt', 'a') as f:
    f.write(file_name+" : " + description_file_str + "\n \n")

with open(path_trace + file_name, 'w') as file:
    file.write(score_model.__str__())

with open(path_dest + file_name, 'w') as file:
    file.write(score_model.__str__())

print("temps execution", time.time()-time1)
