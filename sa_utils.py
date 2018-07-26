import csv
import numpy as np


def read_csv(filename='data/emojify/emojify_data.csv'):
    phrase = []
    sentiment = []

    with open(filename) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)

        i = 0
        for row in csv_reader:
            if i == 0:
                print(row)
                i += 1
            else:
                print(row[2])
                phrase.append(row[2])
                sentiment.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(sentiment, dtype=int)

    return X, Y


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map