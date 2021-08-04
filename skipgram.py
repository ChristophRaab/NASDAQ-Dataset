import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from keras.utils import np_utils
from keras.preprocessing.sequence import make_sampling_table
import scipy.io as sio
import os


def train(cleaned_tweets, tweets, hashtags, sentiment, source_idx, target_idx):
    # Obtain skipgram embedding only
    # Create feature representation: TFIDF-Variants and skipgram embedding with 1000 dimension and negative sampling
    # Output will be saved to disk
    # get_glove_embedding_matrix(cleaned_tweets)
    # get_skipgram_gensim_embedding_matrix(cleaned_tweets)

    # Sentence Skipgram is the base feature representation of the datatset
    X = get_skipgram_sentence_embedding_matrix(cleaned_tweets)
    create_domain_adaptation_dataset(X, tweets, source_idx, target_idx, sentiment)


def get_skipgram_sentence_embedding_matrix(text, dim=200, batch_size=256, window_size=5, epochs=1):
    print("get_skipgram_sentence_embedding_matrix")
    if os.path.isfile("data/sentqs_skipgram_sentence_embedding.npz"):
        loaded_embedding = np.load("data/sentqs_skipgram_sentence_embedding.npz")
        loaded_embedding = loaded_embedding["embedding"]
        print('Loaded Skipgram embedding.')
        return loaded_embedding
    else:
        text = [''.join(x) for x in text]
        t = Tokenizer()
        t.fit_on_texts(text)
        corpus = t.texts_to_sequences(text)
        # print(corpus)
        V = len(t.word_index)
        step_size = len(corpus) // batch_size
        model = Sequential()
        model.add(Dense(units=dim, input_dim=V, activation="softmax"))
        model.add(Dense(units=V, input_dim=dim, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.summary()

        model.fit(generate_data(corpus, window_size, V), epochs=epochs, steps_per_epoch=step_size)
        # model.save("data/sentqs_full_skigram_arc.h5")
        mlb = MultiLabelBinarizer()
        enc = mlb.fit_transform(corpus)
        emb = enc @ model.get_weights()[0]
        np.savez_compressed("data/sentqs_skipgram_sentence_embedding", embedding=emb)
        return emb


def create_domain_adaptation_dataset(X, tweets, source_idx, target_idx, sentiment):
    Xs = X[source_idx]
    Xt = X[target_idx]
    Ys = sentiment[source_idx]
    Yt = sentiment[target_idx]
    data = [Xs, Ys, Xt, Yt]
    np.savez('data/sentqs_dataset.npz', *data)
    sio.savemat('data/sentqs_dataset.mat', {'Xs': Xs, 'Xt': Xt, 'Ys': Ys, 'Yt': Yt})
    source_tweets = [tweets[i] for i in source_idx]
    target_tweets = [tweets[i] for i in target_idx]

    pd.DataFrame(source_tweets).to_csv("data/sentqs_source_tweets.csv")
    pd.DataFrame(target_tweets).to_csv("data/sentqs_target_tweets.csv")
    return Xs, Ys, Xt, Yt


def generate_data(corpus, window_size, V):
    for words in corpus:
        couples, labels = skipgrams(words, V, window_size, negative_samples=1, shuffle=True,
                                    sampling_table=make_sampling_table(V, sampling_factor=1e-05))
        if couples:
            X, y = zip(*couples)
            X = np_utils.to_categorical(X, V)
            y = np_utils.to_categorical(y, V)
            yield X, y
