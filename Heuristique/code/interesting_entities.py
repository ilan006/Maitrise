"""
fichier comprennant toutes les entités interessante
"""

list_type_question_interesting = ['Where?', 'How much / many?', 'What name / is called?', 'Who?', 'When / What year?']

def interesting_entities(type_question):
        interisting_entities_list = ()
        if type_question == 'Where?':
            interisting_entities_list = ("GPE", "LOC", "FAC", "ORG")
        elif type_question == 'How much / many?':
            interisting_entities_list = ("MONEY", "QUANTITY", "PERCENT", "CARDINAL", "TIME", "DATE", "ORDINAL")
        elif type_question == 'What name / is called?':
            interisting_entities_list = (
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", 'FAC')
        elif type_question == 'Who?':
            interisting_entities_list = ("PERSON", "ORG", "NORP", "GPE", "PRODUCT")
        elif type_question == 'When / What year?':
            interisting_entities_list = ("TIME", "DATE", "EVENT")
        else:
            interisting_entities_list = ("PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT",
                                         "EVENT","WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT",
                                         "MONEY","QUANTITY","ORDINAL","CARDINAL")
        return interisting_entities_list
