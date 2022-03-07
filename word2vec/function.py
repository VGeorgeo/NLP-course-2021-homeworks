import numpy as np


def train(data: str):
    def subsampling(sentence, word, subsample):
        freq = (np.array(corpus) == word).sum() / len(sentence)
        return np.random.uniform(0, 1) < 1 - ((freq / subsample) ** 0.5 + 1) * subsample / freq

    def generate_training_data(corpus, window_size, sampling=False, subsample=0.004):

        # инициализация слов
        word_to_id = dict()
        id_to_word = dict()
        # id_to_freq = dict()

        X, Y = [], []
        # all_freq = 0
        # заполнение словаря
        for sentence in corpus:
            for i, word in enumerate(set(sentence)):
                word_to_id[word] = i
                id_to_word[i] = word
                # freq = ((np.array(sentence) == word).sum()/len(sentence))**0.75
                # all_freq += freq
                # id_to_freq[i] = freq

        # for i in id_to_word.keys():
        #    id_to_freq[i] = id_to_freq[i]/all_freq

        for sentence in corpus:
            for i in range(len(sentence)):
                if (sampling == False) or (subsampling(sentence, sentence[i], subsample) == False):
                    nbr_inds = list(range(max(0, i - window_size), i)) + \
                               list(range(i + 1, min(len(sentence), i + window_size + 1)))
                    for j in nbr_inds:
                        X.append(word_to_id[sentence[i]])
                        Y.append(word_to_id[sentence[j]])

        X = np.array(X)
        X = np.expand_dims(X, axis=0)
        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=0)
        Y_one_hot = np.zeros((len(word_to_id), Y.shape[1]))
        Y_one_hot[Y[0], np.arange(Y.shape[1])] = 1

        return X, Y_one_hot, word_to_id, id_to_word

    def initialization(dict_size, emb_size):

        W_emb = np.random.uniform(-1, 1, (dict_size, emb_size))
        W = np.random.uniform(-1, 1, (dict_size, emb_size))

        parameters = {}
        parameters['W_emb'] = W_emb
        parameters['W'] = W

        return parameters

    def forward_propagation(X, parameters):

        W_emb = parameters['W_emb']
        word_vec = W_emb[X.flatten(), :].T

        W = parameters['W']
        Z = np.dot(W, word_vec)

        softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)

        caches = {}
        caches['inds'] = X
        caches['word_vec'] = word_vec
        caches['W_emb'] = W_emb
        caches['W'] = W
        caches['Z'] = Z

        return softmax_out, caches

    def backward_propagation(Y, softmax_out, caches, parameters, learning_rate, m=False):

        # softmax_backward
        dL_dZ = softmax_out - Y

        # dense_backward
        W = caches['W']
        word_vec = caches['word_vec']
        dL_dW = np.dot(dL_dZ, word_vec.T)
        if m == True:
            dL_dW = dL_dW / dL_dZ.shape[1]
        dL_dword_vec = np.dot(W.T, dL_dZ)

        inds = caches['inds']
        parameters['W_emb'][inds.flatten(), :] -= dL_dword_vec.T * learning_rate
        parameters['W'] -= learning_rate * dL_dW

    def skipgram_model_training(X, Y, dict_size, emb_size, learning_rate, epochs, batch_size=64):

        parameters = initialization(dict_size, emb_size)

        for epoch in range(epochs):
            learning_rate *= 0.9
            batch_inds = list(range(0, X.shape[1], batch_size))
            np.random.shuffle(batch_inds)
            for i in batch_inds:
                X_batch = X[:, i:i + batch_size]
                Y_batch = Y[:, i:i + batch_size]

                softmax_out, caches = forward_propagation(X_batch, parameters)
                backward_propagation(Y_batch, softmax_out, caches, parameters, learning_rate)

        return parameters, learning_rate

    window = 3
    emb_size = 10
    learning_rate = 0.2
    batch_size = 1
    epoch = 75

    corpus = [[word for word in data.split()]]
    X, Y_one_hot, word_to_id, id_to_word = generate_training_data(corpus, window, sampling=False, subsample=0.00001)
    dict_size = len(word_to_id)

    params, learning_rate = skipgram_model_training(X, Y_one_hot, dict_size, emb_size, learning_rate, epoch, batch_size)

    word_to_predict = {}
    for i in range(len(word_to_id)):
        word_to_predict[id_to_word[i]] = params['W_emb'][i, :]

    return word_to_predict
