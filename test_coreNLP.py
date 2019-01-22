import corenlp.client as c
import corenlp
import sys
sys.path.append('./')
sys.path.append('./utils/')
from utils import *
text = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."
question = "When was the most recent Super Bowl hosted in the South Florida/Miami area?"

# We assume that you've downloaded Stanford CoreNLP and defined an environment
# variable $CORENLP_HOME that points to the unzipped directory.
# The code below will launch StanfordCoreNLPServer in the background
# and communicate with the server to annotate the sentence.

with c.CoreNLPClient(annotators="tokenize pos lemma".split()) as client:
  ann = client.annotate(question)
  ann = client.annotate(text)
  ann = client.annotate(text)
  ann = client.annotate(question)


# You can access annotations using ann.
sentence = ann.sentence[0]
# print(sentence.token)

# The corenlp.to_text function is a helper function that
# reconstructs a sentence from tokens.
# assert corenlp.to_text(sentence) == text

# You can access any property within a sentence.

# print(sentence.text)

# Likewise for tokens
# token = sentence.token[0]
# print(token.lemma)

# # Use tokensregex patterns to find who wrote a sentence.
# pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
# matches = client.tokensregex(text, pattern)
# # sentences contains a list with matches for each sentence.
# assert len(matches["sentences"]) == 1
# # length tells you whether or not there are any matches in this
# assert matches["sentences"][0]["length"] == 1
# # You can access matches like most regex groups.
# matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
# matches["sentences"][1]["0"]["1"]["text"] == "Chris"
#
# # Use semgrex patterns to directly find who wrote what.
# pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
# matches = client.semgrex(text, pattern)
# # sentences contains a list with matches for each sentence.
# assert len(matches["sentences"]) == 1
# # length tells you whether or not there are any matches in this
# assert matches["sentences"][0]["length"] == 1
# # You can access matches like most regex groups.
# matches["sentences"][1]["0"]["text"] == "wrote"
# matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
# matches["sentences"][1]["0"]["$object"]["text"] == "sentence"




def ClassifyQuestion(question):
    tokens = question.sentence[0].token
    wh_token_index = None
    for i, token in enumerate(tokens):
        if token.pos in ['WRB', 'WP', 'WDT', 'WP$'] and token.lemma != 'that':
            wh_token_index = i
            break

    if wh_token_index is None:
        return 0

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
        return 1

    if tokens[
        wh_token_index].lemma == 'when' or 'what <- det - year' in edge_lemmas or 'what - nsubj -> year' in edge_lemmas:
        return 2

    for wh_token_type in ['what', 'which']:
        for edge_type in [' <- det - ', ' - nsubj -> ', ' <- nsubj - ', ' <- dobj - ', ' - dep -> ', ' <- nmod - ']:
            for noun_type in ['NN', 'NNS', 'NNP']:
                if wh_token_type + edge_type + noun_type in edge_types:
                    return 3

    for edge_type in [' <- dobj - ', ' <- nsubjpass - ', ' <- nsubj - ', ' - nsubj -> ', ' - dep -> ']:
        for verb_type in ['VB', 'VBN', 'VBZ', 'VBP', 'VBD', 'VBG']:
            if 'what' + edge_type + verb_type in edge_types:
                return 4

    if 'how <- advmod - many' in edge_lemmas or 'how <- advmod - much' in edge_lemmas:
        return 5

    if tokens[wh_token_index].lemma == 'how':
        return 6

    if tokens[wh_token_index].lemma in ['who', 'whom', 'whose']:
        return 7

    if tokens[wh_token_index].lemma == 'where':
        return 8

    return 0

print(ClassifyQuestion(ann))