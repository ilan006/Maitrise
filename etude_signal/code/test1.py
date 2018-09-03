import fasttext

# Skipgram model
model = fasttext.skipgram('data.txt', 'model')
print(model.words) # list of words in dictionary

# CBOW model
model = fasttext.cbow('data.txt', 'model')
print(model.words) # list of words in dictionary