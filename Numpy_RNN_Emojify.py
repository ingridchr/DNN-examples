# import numpy as np
from emo_utils import *
# import emoji


# RNN: Vector representation of each word --> Average vector --> SOFTMAX: Prob Vector --> Predicted Class
def __init__():
    # Load dataset: train_emoji (phrase, emoji)
    X_train, Y_train = read_csv('dataset/emojify/train_emoji.csv')

    # Load a pre-trained 50-dimensional GloVe embeddings
    # word_to_index: dict words to indices in the vocabulary (400K)
    # index_to_word: dict indices to words in the vocabulary
    # word_to_vec_map: dict words to their GloVe vector representation
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('dataset/emojify/glove.6B.50d.txt')

    # Train the model to learn the SOFTMAX parameters
    pred, W, b = model(X_train, Y_train, word_to_vec_map)

    print
    print("Training set:")
    pred = predict(X_train, Y_train, W, b, word_to_vec_map)
    print("Accuracy: " + str(np.mean((pred[:] == Y_train.reshape(Y_train.shape[0], 1)[:]))))

    print
    print('Test set:')
    X_test, Y_test = read_csv('dataset/emojify/tesss.csv')
    pred = predict(X_test, Y_test, W, b, word_to_vec_map)
    print("Accuracy: " + str(np.mean((pred[:] == Y_test.reshape(Y_test.shape[0], 1)[:]))))


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    # Split sentence into list of lower case words
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros((50,))

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)

    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    # Define number of training examples
    m = Y.shape[0]  # number of training examples
    n_y = 5  # number of classes
    n_h = 50  # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples

            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = -np.sum(np.multiply(Y_oh[i], np.log(a)))

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)
            print("Accuracy: " + str(np.mean((pred[:] == Y.reshape(Y.shape[0], 1)[:]))))

    return pred, W, b


__init__()

