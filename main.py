#!/usr/bin/env python3

__author__ = "Will Green - UP853829"
__copyright__ = "Copyright 2021, @ME"
__credits__ = ["Will Green"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Will Green"
__email__ = "up853829@myport.ac.uk"
__status__ = "Testing"

# libraries
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Files
from src.Word2Vec_SkipGram import Word2VecSkipGram, architecture
from src.Utils import *

path_SF_North_India_University = open("DataSets/SF-North_India_University.csv")
path_student_survey = open("DataSets/STUDENT-SURVEY.csv")

df = pd.concat(map(pd.read_csv, [path_SF_North_India_University, path_student_survey])).drop_duplicates().isnull()

def parse_sample_data(corpus):
    return pd.read_excel(corpus).drop_duplicates().isnull()

def main():
    word2Vec_skipgram_model = Word2VecSkipGram()
    word2Vec_skipgram_model.network_train(
            word2Vec_skipgram_model.create_sample_data(
            architecture, df))
            
if __name__ == '__main__':
    main()
