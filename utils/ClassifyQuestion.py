import sys

import corenlp.client as c
client = c.CoreNLPClient(annotators="tokenize ssplit pos lemma parse".split())