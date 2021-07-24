import matplotlib.pyplot as plt
import numpy as np
import collections.abc
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

from .Utils import *

architecture = {
    'word_dimension': 5,
    'panel_size': 2,
    'count': int,
    'passing_epoch_value': 674,
    'neg_samp': 10,
    'learning_rate': 0.1,
    'network_seed': np.random.seed(73)
}

class Word2VecSkipGram(object):
    network_loss_rate: int
    vector_occurrence_freq: int
    words_list: list
    word_index: dict
    index_word: dict

    embedding_matrix: np.ndarray
    context_matrix: np.ndarray

    def __init__(self) -> None:
        self.id = architecture['word_dimension']
        self.learning_rate = architecture['learning_rate']
        self.passing_epoch_value = architecture['passing_epoch_value']
        self.panel = architecture['panel_size']

    def create_sample_data(self, architecture: dict, sample_corpus: collections.abc.Iterable) -> list:
        architecture['count'] = 0

        word_counts = defaultdict(int)
        for row in sample_corpus:
            for word in row:
                word_counts[word] += 1

        self.vector_occurrence_freq = len(list(word_counts.keys()))
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, index) for index, word in enumerate(self.words_list))
        self.index_word = dict((index, word) for index, word in enumerate(self.words_list))

        sample_data = []
        for sample_sentence in sample_corpus:
            for i, word in enumerate(sample_sentence):
                target_word = self.oneHot_encode(sample_sentence[i])

                list_word_context = []
                for j in range(i - self.panel, i + self.panel + 1):
                    if j != i and len(sample_sentence) - 1 >= j >= 0:
                        list_word_context.append(self.oneHot_encode(sample_sentence[j]))
                sample_data.append([target_word, list_word_context])
        return sample_data

    def oneHot_encode(self, phonemes: object) -> list:
        ordinal_nominal_column_vectors = []
        for i in range(0, self.vector_occurrence_freq):
            ordinal_nominal_column_vectors.append(0)

        ordinal_nominal_column_vectors[self.word_index[phonemes]] = 1
        return ordinal_nominal_column_vectors

    def propagate_forward(self, node):
        row_vector_transformed = np.dot(self.embedding_matrix.T, node)
        propagation = np.dot(self.context_matrix.T, row_vector_transformed)
        output_index = softmax_function(propagation)
        return output_index, row_vector_transformed, propagation

    def propagate_backward(self, layer_error, row_vector_transformed, node):
        partial_diff_layer_one = np.outer(row_vector_transformed, layer_error)
        partial_diff_layer_two = np.outer(node, np.dot(self.context_matrix, layer_error.T))

        self.embedding_matrix = self.embedding_matrix - (self.learning_rate * partial_diff_layer_two)
        self.context_matrix = self.context_matrix - (self.learning_rate * partial_diff_layer_one)

    def network_train(self, sample_data):
        self.embedding_matrix = np.random.uniform(-0.1, 0.1, (self.vector_occurrence_freq, self.id))
        self.context_matrix = np.random.uniform(-0.1, 0.1, (self.id, self.vector_occurrence_freq))

        for i in range(0, self.passing_epoch_value):
            self.network_loss_rate = 0

            for center_word, word_column_index in sample_data:
                output_prediction, row_vector_transformed, propagation = self.propagate_forward(center_word)

                row_wise_sum = np.sum([np.subtract(output_prediction, word) for word in word_column_index], axis=0)

                self.propagate_backward(row_wise_sum, row_vector_transformed, center_word)
                self.network_loss_rate += -np.sum(
                    [
                        propagation[word.index(1)] for word in word_column_index
                    ]) + len(word_column_index) * np.log(np.sum(np.exp(propagation)))
            print("Epoch value: {} and Loss value: {}".format(i, self.network_loss_rate))

            # For Testing and visualization
            # plt.plot(self.embedding_matrix, "go")
            # plt.plot(self.context_matrix, "ro")
            # plt.xlabel("Word Embedding")
            # plt.ylabel("Vector Weight")
            # plt.show()

    def convert_word_vector(self, phoneme):
        phoneme = self.word_index[phoneme]
        word_vector = self.embedding_matrix[phoneme]
        return word_vector

    def word_vector_sum(self, vector, node):
        word_vector_sum = {}
        for i in range(self.vector_occurrence_freq):
            layer_two_vector = self.embedding_matrix[i]
            bayesian_weighting = np.dot(vector, layer_two_vector)
            bayesian_density = np.linalg.norm(vector) * np.linalg.norm(layer_two_vector)

            word_vector_sum[self.index_word[i]] = (bayesian_weighting / bayesian_density)

        words_sorted = sorted(list(word_vector_sum.items()),
                              key=lambda word_sim1: word_sim1[1],
                              reverse=True)

        # for phoneme, sim in words_sorted[:node]:
        #     print(phoneme, sim)

    def word_sum(self, phoneme, node):
        layer_one_index = self.word_index[phoneme]
        layer_one_vector = self.embedding_matrix[layer_one_index]

        word_sum = {}
        for i in range(self.vector_occurrence_freq):
            layer_two_vector = self.embedding_matrix[i]
            bayesian_weighting = np.dot(layer_one_vector, layer_two_vector)
            bayesian_density = np.linalg.norm(layer_one_vector) * np.linalg.norm(layer_two_vector)

            word_sum[self.index_word[i]] = (bayesian_weighting / bayesian_density)

        words_sorted = sorted(list(word_sum.items()),
                              key=lambda word_sim2: word_sim2[1],
                              reverse=True)

        # for phoneme, sim in words_sorted[:node]:
        #     print(phoneme, sim)
