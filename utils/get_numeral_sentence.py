from nltk import sent_tokenize

def num_sentence(position, text):
    '''
    fonction retournant pour une position donnée (en char) le numero de la phrase correspondante
    :param position: position en char
    :param text: le paragraphe contenant l'ensmeble des phrases
    :return: le numéro de la phrase correspondant à la position donnée
    '''
    tab_sent = sent_tokenize(text)
    taille = 0
    num_line = 0
    for sent in tab_sent:
        taille += len(sent) + 1
        if position <= taille:
            return num_line
        num_line += 1

def pos_relative_sentence(position, text, bool_normalize = True):
    '''
    :param position : la position en char determinant la phrase d'interet
    :param text : le text contenant la phrase d'interet
    :param bool_normalize: booleen indiquant si il faut normaliser la position par la taille du texte
    :return: la postion relative de la phrase dans le paragraphe
    '''
    tab_sent = sent_tokenize(text)
    taille = 0
    num_line = 1
    for sent in tab_sent:
        taille += len(sent) + 1
        if position <= taille:
            return float(num_line) / max(len(tab_sent) * bool_normalize, 1)
        num_line += 1