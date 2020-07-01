# -*- coding: utf-8 -*-
"""
sentqs - NASDAQ Domain Adaptation Dataset

Contens of this file:
Preprocessing
Feature representation
Plotting and description of datasets

authors:  Christoph Raab
"""
import scipy.io as sio
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import cleanup
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
#from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.utils import np_utils
from tensorflow.keras.preprocessing.sequence import skipgrams, pad_sequences
from sklearn.manifold import TSNE
from tensorflow.keras.models import Sequential
from sklearn import decomposition
#from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import make_sampling_table
from numpy import asarray, zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling3D, GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2D, MaxPooling3D, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model
from scipy import spatial
from gensim.models import Word2Vec
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend
import tensorflow as tf

def load_data_run_classification():
    data = np.load('data/sentqs_dataset.npz')
    Xs = data["arr_0"]
    Ys = data["arr_1"]
    Xt = data["arr_2"]
    Yt = data["arr_3"]
    print("Classification Task Test \n")
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(Xs, Ys)
    print(clf.score(Xt, Yt))

def create_tfidf(sen,min_df=10,max_df=100):
    print("Create TF-IDF\n")
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(sen)
    return X.toarray()

def save_dataset(X, y, prefix =""):
    print("Save Dataset \n")
    y = np.array(y)[:, None]
    dataset = np.concatenate([y,X], axis=1)

    np.save("data/sentqs_da_"+str(prefix)+".npy",dataset)
    return dataset

def seperate_tweets(data,hashtags,sentiment):
    print("Seperate Tweets \n")
    labels = []
    tweets = []
    sentiment_new = []
    for t,s in zip(data,sentiment):
        for h in hashtags:
            t = t.lower()
            h = "#" + h.lower()
            if h in t:
                labels.append(h)
                tweets.append(t.replace(h," "))
                sentiment_new.append(s)
                break

    return labels,tweets,sentiment_new

def generate_data(corpus, window_size, V):
    for words in corpus:
        couples, labels = skipgrams(words, V, window_size, negative_samples=1, shuffle=True,sampling_table=make_sampling_table(V, sampling_factor=1e-05))
        if couples:
            X, y = zip(*couples)
            X = np_utils.to_categorical(X, V)
            y = np_utils.to_categorical(y, V)
            yield X, y

def get_glove_embedding_matrix(word_index, dim):
    if os.path.isfile("data/sentqs_glove_embedding.npz"):
        loaded_embedding = np.load("data/sentqs_glove_embedding.npz")
        print('Loaded Glove embedding.')
        return loaded_embedding['embedding']
    else:
        # first, build index mapping words in the embeddings set
        # to their embedding vector

        print('Indexing word vectors.')

        embeddings_index = {}
        with open('data/glove.twitter.27B.200d.txt', encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))
        print('Preparing embedding matrix.')

        # prepare embedding matrix
        num_words = len(word_index) + 1
        embedding_matrix = np.zeros((num_words, dim))
        counter = 0
        for word, i in word_index.items():
            # if i >= MAX_NUM_WORDS:
            #    counter +=1
            #    continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        np.savez_compressed("data/sentqs_glove_embedding.npz", embedding=embedding_matrix)
        return embedding_matrix

def get_skipgram_sentence_embedding_matrix(text, dim=200, batch_size=256, window_size=5, epochs = 100):
    if os.path.isfile("data/sentqs_skipgram_embedding.npz"):
        loaded_embedding = np.load("data/sentqs_skipgram_embedding.npz")
        loaded_embedding = loaded_embedding["embedding"]
        print('Loaded Skipgram embedding.')
        return loaded_embedding
    else:
        text = [''.join(x) for x in text]
        t = Tokenizer()
        t.fit_on_texts(text)
        corpus = t.texts_to_sequences(text)
        print(corpus)
        V = len(t.word_index)
        step_size = len(corpus) // batch_size
        model = Sequential()
        model.add(Dense(dim, input_dim=V, activation="softmax"))
        model.add(Dense(V, input_dim=dim, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.summary()

        model.fit(generate_data(corpus, window_size, V), epochs=epochs, steps_per_epoch=step_size)
        # model.save("data/sentqs_full_skigram_arc.h5")
        mlb = MultiLabelBinarizer()
        enc = mlb.fit_transform(corpus)
        emb = enc @ model.get_weights()[0]
        np.savez_compressed("data/sentqs_skipgram_embedding", embedding=emb)
        return emb

def  get_skipgram_gensim_embedding_matrix(text, dim = 200, window_size=5, min_word_occurance=1, epochs=1):
    if os.path.isfile("data/sentqs_skipgram_gensim_embedding.npz"):
        loaded_embedding = np.load("data/sentqs_skipgram_gensim_embedding.npz")
        loaded_embedding = loaded_embedding["embedding"]
        print('Loaded Skipgram embedding.')
        return loaded_embedding
    else:
        x = [row.split(' ') for row in text]
        model = Word2Vec(x, size=dim, window=window_size, min_count=min_word_occurance, workers=4, sg=1) #sg = 1: use skipgram

        words = model.wv.vocab.keys()
        vocab_size = len(words)
        print("Vocab size", vocab_size)

        t = Tokenizer()
        t.fit_on_texts(text)

        # total vocabulary size plus 0 for unknown words
        # vocab_size = len(vocab) + 1
        # define weight matrix dimensions with all 0
        weight_matrix = zeros((vocab_size, dim))
        # step vocab, store vectors using the Tokenizer's integer mapping
        for word, i in t.word_index.items():
            if i > vocab_size: break
            if word in model.wv.vocab.keys():
                weight_matrix[i] = model.wv[word]

        np.savez_compressed("data/sentqs_skipgram_gensim_embedding", embedding=weight_matrix)
        return weight_matrix

def generate_embedding_model(text, y,source_idx,target_idx,batch_size=32, epochs = 50, save = True, dim = 200, val_split=0.2,model_size="large"):
    # Preprocessing
    #MAX_SEQUENCE_LENGTH = len(max(text, key=lambda i: len(i))) + 1
    MAX_SEQUENCE_LENGTH = 335
    texts = [''.join(x) for x in text]
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(y))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    num_words = len(word_index) + 1

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(val_split * data.shape[0])

    x_train = data[source_idx]
    y_train = labels[source_idx]
    x_val = data[target_idx]
    y_val = labels[target_idx]

    emb = get_skipgram_gensim_embedding_matrix(text, epochs=1)
    emb = np.expand_dims(emb, 1)
    emb_train = emb[:-num_validation_samples]
    emb_val = emb[-num_validation_samples:]
    # Build model
    MAX_SEQUENCE_LENGTH = len(max(text, key=lambda i: len(i))) + 1
    with tf.device('/GPU:0'):
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name="embedding_input")

        glove_embedding_layer = Embedding(num_words,
                                    dim,
                                    weights=[get_glove_embedding_matrix(word_index, dim)],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)(sequence_input)
        skipgram_embedding_layer = Embedding(num_words,
                                    dim,
                                    weights=[get_skipgram_gensim_embedding_matrix(texts, dim)],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)(sequence_input)

        if model_size =="medium":

            combined=  tf.keras.layers.Lambda(lambda t: tf.stack(t,axis=3))([skipgram_embedding_layer, glove_embedding_layer])
            x = Conv2D(128, 5, activation='relu')(combined)
            x = MaxPooling2D(5)(x)
            x = Conv2D(128, 5, activation='relu')(x)
            x = MaxPooling2D(5)(x)
            x = Conv2D(128, 5, activation='relu')(x)
            x = GlobalMaxPooling2D()(x)
            x = Dense(128, activation='relu')(x)



        if model_size =="large":
            own_embedding_layer = Embedding(num_words,
                                    dim,
                                    #embeddings_initializer=Constant(get_skipgram_gensim_embedding_matrix(text, epochs=1)),
                                    #weights=get_skipgram_gensim_embedding_matrix(text, epochs=1),
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)(sequence_input)

            combined = tf.stack([glove_embedding_layer, skipgram_embedding_layer, own_embedding_layer],axis=3)
            x = DenseNet121(include_top=False, weights=None, input_shape = (MAX_SEQUENCE_LENGTH, dim, 3))(combined)
            x = GlobalAveragePooling2D()(x)

        preds = Dense(3, activation='softmax')(x)
        model = Model(inputs=sequence_input, outputs=preds)
        model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['acc'])

        model.summary()
        # plot_model(model, to_file='model_combined.png')

        # Train model
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val))

        if save:
            model.save("data/sentqs_full.h5")
        return model

def tsne_embedding(X):
    print("Starting TSNE\n")
    for p in [5,25,50,75,100]:
        tsne = TSNE(n_components=2, init='random',
             random_state=0, perplexity=p)
        xl = tsne.fit_transform(X)
        np.save("data/sentqs_tsne_"+str(p)+".npy",xl)
        print("Finished TSNE\n")


def describe_dataset(tweets,labels):
    data = pd.DataFrame([tweets, labels]).T
    description = data.describe()
    print(description)

    print("Class Counts:")
    class_counts = data.groupby(1).size()

    x = class_counts.to_numpy()
    keys = class_counts.keys().to_list()
    fig, ax = plt.subplots()
    plt.bar(keys, x)
    plt.ylabel("Tweet Count")
    plt.xticks(range(len(keys)), keys, rotation=45)
    plt.xlabel("Hastags")
    plt.tight_layout()
    plt.savefig("plots/sentqs_class_dist.pdf", dpi=1000, transparent=True)
    plt.show()

def plot_eigenspectrum(x):
    values = np.linalg.svd(x,compute_uv=False)
    plt.bar(range(101), values[:101], align='center')
    plt.ylabel("Eigenvalue")
    # plt.tight_layout()
    plt.xlabel("No.")
    plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.savefig("plots/sentqs_spectra.pdf", transparent=True)
    plt.show()


def plot_tsne(X:None,labels):
    tsne_embedding(X)

    y = preprocessing.LabelEncoder().fit_transform(labels)
    for p in [5, 25, 50, 75, 100]:
        d = np.load("data/sentqs_tsne_" + str(p) + ".npy")
        for idx, l in enumerate(list(set(labels))):
            c = np.where(y == idx)[0]
            x = d[c, :]
            plt.scatter(x[:, 0], x[:, 1], s=.5, label=l)
            plt.legend(markerscale=10., bbox_to_anchor=(1, 1.02))
        plt.ylabel("$x_1$")
        plt.xlabel("$x_2$")
        plt.tight_layout()
        plt.savefig('plots/sentqs_tsne_plot_' + str(p) + ".pdf", dpi=1000, transparent=True)
        plt.show()

def create_domain_adaptation_dataset(X,tweets,source_idx,target_idx,sentiment):

    Xs = X[source_idx]
    Xt = X[target_idx]
    Ys = sentiment[source_idx]
    Yt = sentiment[target_idx]
    data = [Xs,Ys,Xt,Yt]
    np.savez('data/sentqs_dataset.npz', *data)
    sio.savemat('data/sentqs_dataset.mat', {'Xs': Xs, 'Xt': Xt, 'Ys': Ys, 'Yt': Yt})
    source_tweets = [tweets[i] for i in source_idx]
    target_tweets = [tweets[i] for i in target_idx]

    pd.DataFrame(source_tweets).to_csv("data/sentqs_source_tweets.csv")
    pd.DataFrame(target_tweets).to_csv("data/sentqs_target_tweets.csv")
    return  Xs,Ys,Xt,Yt

def load_sentqs_tweets():
    if os.path.isfile("data/sentqs_preprocessed.npz"):
        loaded_data = np.load("data/sentqs_preprocessed.npz")
        return loaded_data['cleaned_tweets'],loaded_data["tweets"], loaded_data['y'],loaded_data['sentiment'],loaded_data["source_idx"],loaded_data["target_idx"]
    else:
        hashtags = ['ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG', 'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX',
         'TSLA', 'XEL', 'positive', 'bad', 'sad']

        # Loading and preprocessing of tweets
        df = pd.read_csv("Tweets.csv")
        sentiment = pd.to_numeric(df.iloc[:, -1], errors="raise", downcast="float")
        labels,tweets,sentiment = seperate_tweets(df.iloc[:, 1],hashtags,sentiment)
        cleaned_tweets = cleanup.clean_text(tweets)

        y = preprocessing.LabelEncoder().fit_transform(labels)

        source_idx,target_idx = create_domain_adaptation_index(tweets,labels,sentiment)
        np.savez_compressed("data/sentqs_preprocessed.npz",tweets=tweets, cleaned_tweets=cleaned_tweets, y=y,sentiment=sentiment,source_idx=source_idx,target_idx=target_idx)
        return cleaned_tweets,tweets, y,sentiment,source_idx,target_idx

def create_domain_adaptation_index(tweets,labels,sentiment):
    labels = np.array([s if "#bad" not in s else "#sad" for s in labels])
    source_idx  = np.array([i for i,val in enumerate(labels) if val== "#sad" or val == "#bad" ])
    target_idx = np.array([i for i,val in enumerate(labels) if val != "#sad" and val != "#bad" ])
    return source_idx,target_idx


def main_preprocessing():

    cleaned_tweets,tweets,hashtags,sentiment, source_idx, target_idx = load_sentqs_tweets()

    # Obtain embeddings and train deep learning model
    # model = generate_embedding_model(cleaned_tweets,sentiment,source_idx,target_idx,model_size="large")


    # Obtain skipgram embedding only
    # Create feature representation: TFIDF-Variants and skipgram embedding with 1000 dimension and negative sampling
    # # Output will be saved to disk
    # X = get_skipgram_sentence_embedding_matrix(cleaned_tweets)
    # create_domain_adaptation_dataset(X,tweets,source_idx,target_idx,sentiment)
    # # Another possible embedding:
    # # get_skipgram_gensim_embedding_matrix(cleaned_tweets)



    # # Describe dataset with some common characteristics
    # describe_dataset(cleaned_tweets,hashtags)

    # ## Plot eigenspectrum of embeddings
    # X = np.load("data/sentqs_skipgram_sentence_embedding.npz",allow_pickle=True)
    # plot_eigenspectrum(X)

    # ## Plot representation of 2 dimensional tsne embedding
    # plot_tsne(X,sentiment)

    ## Loads the data into the program and trains machine learning model
    load_data_run_classification()


if __name__ == '__main__':

    # Obtain the all files of the dataset preprocessing, including plots, feature representation etc.
    # After running this file you will find the corresponding files for classification in the data folder
    main_preprocessing()