from nltk import sent_tokenize


# retourne le numero de la phrase pour une position donné dans un text (en char)
def numeral_sentence(position, text, bool_normalize = True):
    tab_sent = sent_tokenize(text)
    taille = 0
    num_line = 1
    for sent in tab_sent:
        taille += len(sent) + 1
        if position <= taille:
            return float(num_line) / max(len(tab_sent) * bool_normalize, 1)
        num_line += 1

