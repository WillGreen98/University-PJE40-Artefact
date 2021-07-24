import re
import numpy as np
from nltk.corpus import stopwords
import string

def softmax_function(arg):
    return np.exp(arg - np.max(arg)) / np.exp(arg - np.max(arg)).sum(axis=0)

def data_preprocess(corpus):
    corpus = open("DataSets/SF-North_India_University.csv")
    sample_data = []

    sentences = corpus.readline().split(".")
    stop_words = set(stopwords.words('english'))

    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence if word not in stop_words]
        x = [word.lower() for word in x]
        sample_data.append(x)
    return sample_data
