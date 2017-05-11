from collections import Counter
import json
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import random
import string

def get_classification():
	types = {"normal": [1,0], "minor": [1,0], "trivial": [1,0], "major": [0,1], "critical": [0,1]}
	return types

def create_feature_vector(sample, lexicon=None):
	if lexicon is None:
		with open("../data/firefox-defect-criticality_lexicon.json") as infile:
			lexicon = json.load(infile)
	features = np.zeros(len(lexicon))
	words = word_tokenize(sample.lower())
	for word in words:
		if word.lower() in lexicon:
			features[lexicon.index(word.lower())] += 1
	return list(features)

def create_feature_sets(filename, test_size=0.2):
	classification = {}
	df = pd.read_csv(filename, low_memory=False, encoding="ISO-8859-1")
	features = []
	lexicon = create_lexicon(df)
	classification = get_classification()
	features = sample_handling(df, lexicon, classification)
	random.shuffle(features)
	features = np.array(features)
	testing_size = int(test_size * len(features))
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])
	return train_x, train_y, test_x, test_y

def create_lexicon(df):
	lexicon = []
	stop = set(stopwords.words('english') + list(string.punctuation))
	stop.add("quot")
	for index, row in df.iterrows():
		for i in word_tokenize(str(row["Title/Description"]).lower()):
			if not is_stopword(i):
				lexicon.append(i)
	word_counts = Counter(lexicon)
	counts = np.array(list(word_counts.values()))
	r1 = np.percentile(counts, 75)
	lexicon = [word for word in word_counts if r1 <= word_counts[word]]
	with open('../data/firefox-defect-criticality_lexicon.json', 'w') as outfile:
		json.dump(lexicon, outfile)
	return lexicon

def is_stopword(word):
	if len(word) == 1:
		return True
	stop = set(stopwords.words('english') + list(string.punctuation))
	stop.add("quot")
	if word in stop:
		return True
	alpha = 0
	non_alpha = 0
	for each in word:
		if each.isalpha():
			alpha += 1
		else:
			non_alpha += 1
	if non_alpha > alpha:
		return True
	return False

def sample_handling(sample, lexicon, classification):
	feature_set = []
	i = 0
	for index, row in sample.iterrows():
		defect_type = str(row["Importance/Type"]).lower()
		if defect_type in classification:
			features = create_feature_vector(str(row["Title/Description"]), lexicon)
			feature_set.append([features, classification[defect_type]])
	return feature_set

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_feature_sets("../data/firefox-dataset.csv")
	with open('../data/firefox-defect-criticality_set.json', 'w') as outfile:
		json.dump([train_x, train_y, test_x, test_y], outfile)