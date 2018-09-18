import corenlp.client as c
import subprocess
client = c.CoreNLPClient(annotators="tokenize ssplit pos lemma parse".split())


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
question = "What is your name?"

subprocess.Popen("java")



print(ClassifyQuestion(question))
# print(ClassifyQuestion(question))
# print(ClassifyQuestion(question))
# print(ClassifyQuestion(question))

