from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from sa_utils import *
from clean_data import get_train_test_dataset


def __init__():
    # Load a pre-trained 50-dimensional GloVe embeddings
    # word_to_index: dict words to indices in the vocabulary (400K)
    # index_to_word: dict indices to words in the vocabulary
    # word_to_vec_map: dict words to their GloVe vector representation
    print
    print("Reading GloVe Vectors")
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('dataset/emojify/glove.6B.50d.txt')
    print("Reading Training/Test Set")
    X_train, X_test, Y_train, Y_test = get_train_test_dataset('dataset/sentiment_analysis/train.csv')

    # print("ACA" , max(X_train, key=len).split())

    maxLen = len(max(X_train, key=len).split())
    print(maxLen)

    model = keras_model((maxLen,), word_to_vec_map, word_to_index)
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)

    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)

    # # This code allows you to see the mislabelled examples
    # C = 5
    # y_test_oh = np.eye(C)[Y_test.reshape(-1)]
    # X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    # pred = model.predict(X_test_indices)
    # for i in range(len(X_test)):
    #     x = X_test_indices
    #     num = np.argmax(pred[i])
    #     if (num != Y_test[i]):
    #         print('Expected emoji:' + label_to_emoji(Y_test[i]) + ' prediction: ' + X_test[i] + label_to_emoji(
    #             num).strip())
    #
    # # Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.
    # x_test = np.array(['I admire you'])
    # X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    # print(x_test[0] + ' ' + label_to_emoji(np.argmax(model.predict(X_test_indices))))


def keras_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype
    # 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    return model


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        print(sentence_words)

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["people"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    # Initialize the embedding matrix as a array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Build the embedding layer.
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


__init__()