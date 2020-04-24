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
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
#from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model
from scipy import spatial

def run_classification():
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

def seperate_tweets(data,hashtags):
    print("Seperate Tweets \n")
    labels = []
    tweets = []
    for t in data:
        for h in hashtags:
            t = t.lower()
            h = "#" + h.lower()
            if h in t:
                labels.append(h)
                tweets.append(t.replace(h," "))
                break

    return labels,tweets

def generate_data(corpus, window_size, V):
    for words in corpus:
        couples, labels = skipgrams(words, V, window_size, negative_samples=1, shuffle=True,sampling_table=make_sampling_table(V, sampling_factor=1e-05))
        if couples:
            X, y = zip(*couples)
            X = np_utils.to_categorical(X, V)
            y = np_utils.to_categorical(y, V)
            yield X, y

# def create_embedding(text,dim=200,batch_size=256,window_size=5,epochs = 100):
#     text = [''.join(x) for x in text]
#     t = Tokenizer()
#     t.fit_on_texts(text)
#     corpus = t.texts_to_sequences(text)
#     print(corpus)
#     V = len(t.word_index)
#     step_size = len(corpus) // batch_size
#     model = Sequential()
#     model.add(Dense(input_dim=V, output_dim=dim,activation="softmax"))
#     model.add(Dense(input_dim=dim, output_dim=V, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#     model.summary()
#
#     model.fit_generator(generate_data(corpus,window_size,V),epochs=epochs,steps_per_epoch=step_size)
#     #model.save("data/sentqs_full_skigram_arc.h5")
#     mlb = MultiLabelBinarizer()
#     enc = mlb.fit_transform(corpus)
#     emb = enc @ model.get_weights()[0]
#     #np.save("data/sentqs_skipgram_embedding.npy", emb)
#     return emb, model.get_weights()[0]
#
# def create_glove_embedding(text, y, dim=200, batch_size=256, epochs = 100, val_split=0.2, MAX_NUM_WORDS = 20000):
#     MAX_SEQUENCE_LENGTH = len(max(text, key=lambda i: len(i))) + 1
#
#     # first, build index mapping words in the embeddings set
#     # to their embedding vector
#
#     print('Indexing word vectors.')
#
#     embeddings_index = {}
#     with open('data/glove.twitter.27B.200d.txt', encoding="utf8") as f:
#         for line in f:
#             word, coefs = line.split(maxsplit=1)
#             coefs = np.fromstring(coefs, 'f', sep=' ')
#             embeddings_index[word] = coefs
#
#     print('Found %s word vectors.' % len(embeddings_index))
#
#     texts = [''.join(x) for x in text]
#     # finally, vectorize the text samples into a 2D integer tensor
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(texts)
#     sequences = tokenizer.texts_to_sequences(texts)
#
#     word_index = tokenizer.word_index
#     print('Found %s unique tokens.' % len(word_index))
#
#     data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#
#     labels = to_categorical(np.asarray(y))
#     print('Shape of data tensor:', data.shape)
#     print('Shape of label tensor:', labels.shape)
#
#     # split the data into a training set and a validation set
#     indices = np.arange(data.shape[0])
#     np.random.shuffle(indices)
#     data = data[indices]
#     labels = labels[indices]
#     num_validation_samples = int(val_split * data.shape[0])
#
#     x_train = data[:-num_validation_samples]
#     y_train = labels[:-num_validation_samples]
#     x_val = data[-num_validation_samples:]
#     y_val = labels[-num_validation_samples:]
#
#     print('Preparing embedding matrix.')
#
#     # prepare embedding matrix
#     num_words = len(word_index) + 1
#     embedding_matrix = np.zeros((num_words, dim))
#     counter=0
#     for word, i in word_index.items():
#         #if i >= MAX_NUM_WORDS:
#         #    counter +=1
#         #    continue
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#
#
#     # load pre-trained word embeddings into an Embedding layer
#     # note that we set trainable = False so as to keep the embeddings fixed
#     embedding_layer = Embedding(num_words,
#                                 dim,
#                                 embeddings_initializer=Constant(embedding_matrix),
#                                 input_length=MAX_SEQUENCE_LENGTH,
#                                 trainable=False)
#
#     print('Training model.')
#
#     # train a 1D convnet with global maxpooling
#     sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences = embedding_layer(sequence_input)
#     x = Conv1D(128, 5, activation='relu')(embedded_sequences)
#     x = MaxPooling1D(5)(x)
#     x = Conv1D(128, 5, activation='relu')(x)
#     x = MaxPooling1D(5)(x)
#     x = Conv1D(128, 5, activation='relu')(x)
#     x = GlobalMaxPooling1D()(x)
#     x = Dense(128, activation='relu')(x)
#     preds = Dense(18, activation='softmax')(x)
#
#     model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['acc'])
#
#     model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(x_val, y_val))
#
#     model.save("data/sentqs_full_glove.h5")
#
#     mlb = MultiLabelBinarizer()
#     enc = mlb.fit_transform(sequences)
#     emb = enc @ model.get_weights()[1][:-1]
#     np.save("data/sentqs_glove_embedding.npy", emb)
#     print(emb.shape)
#     print(model.get_weights()[0].shape)
#     return emb, model.get_weights()[0]

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

def get_skipgram_embedding_matrix(text, dim=200, batch_size=256, window_size=5, epochs = 100):
    if os.path.isfile("data/sentqs_skipgram.npy"):
        loaded_embedding = np.load("data/sentqs_skipgram.npy")
        print('Loaded Skipgram embedding.')
        return loaded_embedding[:,1:]
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

        model.fit_generator(generate_data(corpus, window_size, V), epochs=epochs, steps_per_epoch=step_size)
        # model.save("data/sentqs_full_skigram_arc.h5")
        mlb = MultiLabelBinarizer()
        enc = mlb.fit_transform(corpus)
        emb = enc @ model.get_weights()[0]
        np.savez_compressed("data/sentqs_skipgram_embedding.npy", embedding=emb)
        return emb

def generate_embedding_model(text, y, batch_size=256, epochs = 100, save = True, dim = 200, val_split=0.2):
    # Preprocessing
    MAX_SEQUENCE_LENGTH = len(max(text, key=lambda i: len(i))) + 1
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

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    emb = get_skipgram_embedding_matrix(text, epochs=1)
    emb = np.expand_dims(emb, 1)
    emb_train = emb[:-num_validation_samples]
    emb_val = emb[:-num_validation_samples]
    # Build model
    MAX_SEQUENCE_LENGTH = len(max(text, key=lambda i: len(i))) + 1

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name="embedding_input")

    glove_embedding_layer = Embedding(num_words,
                                dim,
                                #embeddings_initializer=Constant(get_glove_embedding_matrix(word_index, dim)),
                                weights=[get_glove_embedding_matrix(word_index, dim)],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(sequence_input)

    skipgram_embedding = Input(shape=(1,dim,),name="skipgram_input")


    own_embedding_layer = Embedding(num_words,
                                dim,
                                #embeddings_initializer=Constant(get_skipgram_embedding_matrix(text, epochs=1)),
                                #weights=get_skipgram_embedding_matrix(text, epochs=1),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)(sequence_input)

    combined = concatenate([glove_embedding_layer, skipgram_embedding, own_embedding_layer],axis=1)

    x = Conv1D(128, 5, activation='relu')(combined)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(18, activation='softmax')(x)

    model = Model(inputs=[sequence_input,skipgram_embedding], outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
    # plot_model(model, to_file='model_combined.png')

    # Train model
    model.fit([x_train,emb_train], y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([x_val,emb_val], y_val))

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

# def create_representation(cleaned_tweets, y):
#     #X,weights = create_embedding(cleaned_tweets,dim=200,epochs=1)
#     get_skipgram_embedding_matrix(cleaned_tweets, dim=200, epochs=1)
#
#     #save_dataset(X, y, prefix="skipgram")
#     #X,weights = create_glove_embedding(cleaned_tweets, y, dim=200,epochs=1)
#     #save_dataset(X, y, prefix="glove")
#     #X = create_tfidf(cleaned_tweets)
#     #save_dataset(X, y, prefix="tfidf_small")
#     #X = create_tfidf(cleaned_tweets,min_df=1,max_df=1)
#     #save_dataset(X, y,prefix="tfidf_default")

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

def create_domain_adaptation_problem(X,tweets,labels,sentiment):
    # hastags positive bad sad source und rest target
    labels = np.array([s if "#bad" not in s else "#sad" for s in labels])
    y = preprocessing.LabelEncoder().fit_transform(labels)
    source = np.where(np.logical_or(labels == "#bad", labels == "#sad",labels == "#positive"))[0]
    target = np.where(np.logical_not(labels == "#sad", labels == "#positive"))[0]
    Xs = X[source]
    Xt = X[target]
    Ys = sentiment[source]
    Yt = sentiment[target]
    data = [Xs,Ys,Xt,Yt]
    np.savez('data/sentqs_dataset.npz', *data)
    sio.savemat('data/sentqs_dataset.mat', {'Xs': Xs, 'Xt': Xt, 'Ys': Ys, 'Yt': Yt})
    source_tweets = [tweets[i] for i in source]
    target_tweets = [tweets[i] for i in target]

    pd.DataFrame(source_tweets).to_csv("data/sentqs_source_tweets.csv")
    pd.DataFrame(target_tweets).to_csv("data/sentqs_target_tweets.csv")
    return  Xs,Ys,Xt,Yt

def load_preprocessed_sentqs():
    if os.path.isfile("data/sentqs_preprocessed.npz"):
        loaded_data = np.load("data/sentqs_preprocessed.npz")
        return loaded_data['cleaned_tweets'], loaded_data['y'],loaded_data['sentiment']
    else:
        hashtags = ['ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG', 'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX',
         'TSLA', 'XEL', 'positive', 'bad', 'sad']

        # Loading and preprocessing of tweets
        df = pd.read_csv("Tweets.csv")
        sentiment = pd.to_numeric(df.iloc[:, -1], errors="raise", downcast="float")
        labels,tweets = seperate_tweets(df.iloc[:, 1],hashtags)
        cleaned_tweets = cleanup.clean_text(tweets)

        y = preprocessing.LabelEncoder().fit_transform(labels)
        np.savez_compressed("data/sentqs_preprocessed.npz", cleaned_tweets=cleaned_tweets, y=y,sentiment=sentiment)
        return cleaned_tweets, y,sentiment

def main_preprocessing():

    cleaned_tweets,hashtags,sentiment= load_preprocessed_sentqs()
    #
    # # Get some statistics of the dataset
    # describe_dataset(cleaned_tweets,labels)
    #
    # # Create feature representation: TFIDF Variants and skipgram embedding with 1000 dimension and negative sampling
    get_skipgram_embedding_matrix(cleaned_tweets)
    X = np.load("data/sentqs_skipgram_embedding.npy")
    create_domain_adaptation_problem(X,tweets,hashtags,sentiment)
    # model = generate_embedding_model(cleaned_tweets,sentiment)
    # # Plot eigenspectrum of embeddings
    # X = np.load("data/sentqs_skipgram_embedding.npy")
    # plot_eigenspectrum(X)
    #
    # # Plot representation of 2 dimensional tsne embedding
    # plot_tsne(X,sentiment)
    #
    #X = np.load("data/sentqs_skipgram_embedding.npy")
    #create_domain_adaptation_problem(X,tweets,sentiment,sentiment)

    #run_classification()


if __name__ == '__main__':

    # Obtain the all files of the dataset preprocessing, including plots, feature representation etc.
    # After running this file you will find the corresponding files for classification in the data folder
    main_preprocessing()
