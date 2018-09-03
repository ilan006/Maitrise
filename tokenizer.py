import re


def naiveRegexTokenizer(string, caseSensitive=True, eliminateEnStopwords=False):
	'''
	returns the token list using a very naive regex tokenizer
	'''
	plainWords = re.compile(r'(\b\w+\b)', re.UNICODE)
	tokens = re.findall(plainWords, string)
	#if we don't want to be case sensitive
	if caseSensitive != True:
		tokens = [tok.lower() for tok in tokens]
	#if we don't want the stopwords
	if eliminateEnStopwords != False:
		from nltk.corpus import stopwords
		#stopwords
		to_remove = set(stopwords.words("english") + ['', ' ', '&'])
		tokens = list(filter(lambda tok: tok not in to_remove, tokens))
	return tokens


def tokeniseur(texteASegmenter, motsOuPhrase="mot"):
	# la fonction tokeniseur segmente une txt en un dictionnaire de ses mots, si ceux-ci sont segmentables par les moyens traditionnels, aucune exception n'est prise en compte

	texteASegmenter = re.sub('\d+', '', texteASegmenter)
	texteDejaSegmenteMot = re.findall(r"[\w']+", texteASegmenter)
	# ou texteDejaSegmenteMot = re.compile("\W+", re.UNICODE)

	texteSegmentePhrase = re.compile("[。\.\?\!\n\r\t\;\(\)\:\[\]]+", re.UNICODE)
	texteDejaSegmentePhrase = texteSegmentePhrase.split(texteASegmenter)
	# ou texteDejaSegmentePhrase = enleveurDeMotsVides(re.split(u"[。\.\?\!\;\:\n]+", texteASegmenter))

	motsOuPhrase = str(motsOuPhrase)
	if motsOuPhrase.lower() == "mot" or motsOuPhrase.lower() == "mots" or motsOuPhrase.lower() == "m" or motsOuPhrase == "1":
		return texteDejaSegmenteMot
	if motsOuPhrase.lower() == "phrase" or motsOuPhrase.lower() == "phrases" or motsOuPhrase.lower() == "p" or motsOuPhrase == "2":
		return texteDejaSegmentePhrase