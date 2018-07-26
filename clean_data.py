import csv
import io
import numpy as np


def divide_train_test(array, factor=0.7):

    len_train = int(round(len(array) * factor))

    return array[:len_train], array[len_train:]


def get_train_test_dataset(filename='dataset/sentiment_analysis/train.csv'):
    X, Y = read_csv()
    X_train, X_test = divide_train_test(X)
    Y_train, Y_test = divide_train_test(Y)
    return X_train, X_test, Y_train, Y_test


def read_csv(filename='dataset/sentiment_analysis/train.csv'):
    phrase = []
    sentiment = []

    with open(filename) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        next(csv_reader)  # skip header
        # i = 0
        for row in csv_reader:
            # if i == 99999:
            #     print(i, row[0], row[2])
            # i += 1
            phrase.append(row[2])
            sentiment.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(sentiment, dtype=int)

    return X, Y
