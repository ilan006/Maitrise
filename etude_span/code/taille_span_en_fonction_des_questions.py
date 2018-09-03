import json
import sys
path_data = '../../data/'
path_dest = '../../data_perso/'

sys.path.append("../../")
from nltk import sent_tokenize

dict = {}
nb_answers = 0
total_words = 0

import corenlp.client as c
import corenlp #/u/elbazila/stanford-corenlp-full-2018-01-31/

client = c.CoreNLPClient(annotators="tokenize ssplit pos lemma parse".split())

# 0 = 'Other'
# 1 = 'What name / is called?'
# 2 = 'When / What year?'
# 3 = 'What / Which NN[*]?'
# 4 = 'What VB[*]?'
# 5 = 'How much / many?'
# 6 = 'How?'
# 7 = 'Who?'
# 8 = 'Where?'

def ClassifyQuestion(question):
    question = client.annotate(question)
    tokens = question.sentence[0].token

    wh_token_index = None
    for i, token in enumerate(tokens):
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            wh_token_index = i
            break

    if wh_token_index is None:
        return 'Other'

    edge_types = set()
    edge_lemmas = set()
    for edge in question.sentence[0].basicDependencies.edge:
        if edge.dep in ['punct', 'cop', 'root']:
            continue

        if edge.source == wh_token_index + 1:
            if tokens[edge.target - 1].lemma == 'in':
                continue
            edge_types.add(tokens[wh_token_index].lemma + ' - ' + edge.dep + ' -> ' + tokens[edge.target - 1].pos)
            edge_lemmas.add(tokens[wh_token_index].lemma + ' - ' + edge.dep + ' -> ' + tokens[edge.target - 1].lemma)

        if edge.target == wh_token_index + 1:
            edge_types.add(tokens[wh_token_index].lemma + ' <- ' + edge.dep + ' - ' + tokens[edge.source - 1].pos)
            edge_lemmas.add(tokens[wh_token_index].lemma + ' <- ' + edge.dep + ' - ' + tokens[edge.source - 1].lemma)

    if 'what <- det - name' in edge_lemmas or 'what - nsubj -> name' in edge_lemmas or 'what <- dobj - call' in edge_lemmas or 'what <- nsubjpass - call' in edge_lemmas:
        return 'What name / is called?'

    if tokens[
        wh_token_index].lemma == 'when' or 'what <- det - year' in edge_lemmas or 'what - nsubj -> year' in edge_lemmas:
        return 'When / What year?'

    for wh_token_type in ['what', 'which']:
        for edge_type in [' <- det - ', ' - nsubj -> ', ' <- nsubj - ', ' <- dobj - ', ' - dep -> ', ' <- nmod - ']:
            for noun_type in ['NN', 'NNS', 'NNP']:
                if wh_token_type + edge_type + noun_type in edge_types:
                    return 'What / Which NN[*]?'

    for edge_type in [' <- dobj - ', ' <- nsubjpass - ', ' <- nsubj - ', ' - nsubj -> ', ' - dep -> ']:
        for verb_type in ['VB', 'VBN', 'VBZ', 'VBP', 'VBD', 'VBG']:
            if 'what' + edge_type + verb_type in edge_types:
                return 'What VB[*]?'

    if 'how <- advmod - many' in edge_lemmas or 'how <- advmod - much' in edge_lemmas:
        return 'How much / many?'

    if tokens[wh_token_index].lemma == 'how':
        return 'How?'

    if tokens[wh_token_index].lemma in ['who', 'whom', 'whose']:
        return 'Who?'

    if tokens[wh_token_index].lemma == 'where':
        return 'Where?'

    return 'Other'



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
                        nb_words_ans = len(span.split())
                        total_words += nb_words_ans
                        nb_answers += 1
                        type_question = ClassifyQuestion(question['question'])
                        if type_question in dict:
                            dict[type_question] ["total_ans"] += 1
                            if nb_words_ans in dict[type_question]:
                                dict[type_question][nb_words_ans] += 1
                            else:
                                dict[type_question][nb_words_ans] = 1
                        else:
                            dict[type_question] = {nb_words_ans: 1}
                            dict[type_question] = {"total_ans": 1}
                if nb_answers % 1000 == 0:
                    print(nb_answers)
print(nb_answers)
print(float(total_words)/nb_answers)

with open('../taille_span_dev_en_fonction_des_questions.csv', 'w') as f:
    item_str = "type question" + ',' + "taille" + "," + "compte relatif" + "," + "compte" + "\n"
    f.write(item_str)
    for type_question in dict:
        for taille in dict[type_question]:
            if taille == "total_ans":
                continue
            item_str = str(type_question) + "," + str(taille) + "," + str(dict[type_question][taille]/ float(dict[type_question]["total_ans"])) +"," +str(dict[type_question][taille]) + "\n"
            f.write(item_str)