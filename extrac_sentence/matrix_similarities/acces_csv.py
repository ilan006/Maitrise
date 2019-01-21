from numpy import genfromtxt
import pandas as pd
import time
path_directory = '../../../Data_Maitrise/Matrix_similarite/'# path of files' directory

import os

files = os.listdir(path_directory)
for name in files:
    df = pd.read_csv(path_directory+name, sep=',')
    print(list(df)) #affichage des mots de la question
    print(df) #affichage des deux colones : mots de la phrase, valeur cosine avec le mot de la question 'leader'
    time.sleep(5)