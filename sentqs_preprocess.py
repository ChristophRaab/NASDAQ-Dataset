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
from sklearn import preprocessing
import cleanup
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.sequence import skipgrams
from sklearn.manifold import TSNE
from keras.models import Sequential
from sklearn import decomposition
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import make_sampling_table
from glove import *
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

def create_embedding(text,dim=300,batch_size=256,window_size=5,epochs = 100):
    text = [''.join(x) for x in text]
    t = Tokenizer()
    t.fit_on_texts(text)
    corpus = t.texts_to_sequences(text)
    print(corpus)
    V = len(t.word_index)
    step_size = len(corpus) // batch_size
    model = Sequential()
    model.add(Dense(input_dim=V, output_dim=dim,activation="softmax"))
    model.add(Dense(input_dim=dim, output_dim=V, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()

    model.fit_generator(generate_data(corpus,window_size,V),epochs=epochs,steps_per_epoch=step_size)
    model.save("data/sentqs_full_skigram_arc.h5")
    mlb = MultiLabelBinarizer()
    enc = mlb.fit_transform(corpus)
    emb = enc @ model.get_weights()[0]
    np.save("data/sentqs_skipgram_embedding.npy", emb)
    return emb, model.get_weights()[0]

def create_glove_embedding(corpus,dim=300,batch_size=256,window_size=5,epochs = 100):
    # https://pypi.org/project/glove-py/
    # Print first entries of text/corpus = first 10 tweets
    print(corpus[0:10])

    # Build Vocab from Corpus and show the first 10 words
    vocab = build_vocab(corpus)
    for x in list(vocab)[0:10]:
        print("key: {}, id: {}, frequency: {} ".format(x, vocab[x][0], vocab[x][1]))
    print(len(list(vocab)))

    # Build co-occurrence matrix from the vocab and corpus
    cooccur = build_cooccur(vocab, corpus, window_size)

    # Example of the first 10 word combinations with their co-occurrence value
    for n in range(10):
        occur_example = next(cooccur)
        print("main word: {}, context word: {}, co-occurrence value: {} ".format([word for (word, idfrq) in vocab.items() if idfrq[0] == occur_example[0]], [word for (word, idfrq) in vocab.items() if idfrq[0] == occur_example[1]], occur_example[2]))

    # Train the glove embedding, returns the computed word vector matrix weights
    weights = train_glove(vocab, cooccur, vector_size=dim, iterations=1)
    print(np.array(weights).shape)
    print(weights[0])

    #mlb = MultiLabelBinarizer()
    #enc = mlb.fit_transform(corpus)
    #emb = enc @ weights
    #np.save("data/sentqs_glove_embedding.npy", emb)
    np.save("data/sentqs_glove_weights.np", weights)
    return weights #emb, weights

def tsne_embedding(X):
    print("Starting TSNE\n")
    for p in [5,25,50,75,100]:
        tsne = TSNE(n_components=2, init='random',
             random_state=0, perplexity=p)
        xl = tsne.fit_transform(X)
        np.save("data/sentqs_tsne_"+str(p)+".npy",xl)
        print("Finished TSNE\n")

def create_representation(cleaned_tweets, y):
    #X,weights = create_embedding(cleaned_tweets,dim=300)
    #save_dataset(X, y, prefix="skipgram")
    X,weights = create_glove_embedding(cleaned_tweets,dim=300,epochs=5)
    save_dataset(X, y, prefix="glove")
    #X = create_tfidf(cleaned_tweets)
    #save_dataset(X, y, prefix="tfidf_small")
    #X = create_tfidf(cleaned_tweets,min_df=1,max_df=1)
    #save_dataset(X, y,prefix="tfidf_default")


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
    plt.savefig("data/sentqs_class_dist.pdf", dpi=1000, transparent=True)
    plt.show()

def plot_eigenspectrum(x):
    values = np.linalg.svd(x,compute_uv=False)
    plt.bar(range(101), values[:101], align='center')
    plt.ylabel("Eigenvalue")
    # plt.tight_layout()
    plt.xlabel("No.")
    plt.xticks([0, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.savefig("data/sentqs_spectra.pdf", transparent=True)
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
        plt.savefig('data/sentqs_tsne_plot_' + str(p) + ".pdf", dpi=1000, transparent=True)
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

    pd.DataFrame(source_tweets).to_csv("data/nsdq_source_tweets.csv")
    pd.DataFrame(target_tweets).to_csv("data/nsdq_target_tweets.csv")
    return  Xs,Ys,Xt,Yt



def main_preprocessing():
    hashtags = ['ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG', 'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX',
     'TSLA', 'XEL', 'positive', 'bad', 'sad']

    # Loading and preprocessing of tweets
    df = pd.read_csv("Tweets.csv")
    sentiment = pd.to_numeric(df.iloc[:, -1], errors="raise", downcast="float")
    labels,tweets = seperate_tweets(df.iloc[:, 1],hashtags)
    cleaned_tweets = cleanup.clean_text(tweets)

    y = preprocessing.LabelEncoder().fit_transform(labels)
    #
    # # Get some statistics of the dataset
    # describe_dataset(cleaned_tweets,labels)
    #
    # # Create feature representation: TFIDF Variants and skipgram embedding with 1000 dimension and negative sampling
    create_representation(cleaned_tweets,y)
    #
    # # Plot eigenspectrum of embeddings
    # X = np.load("data/sentqs_skipgram_embedding.npy")
    # plot_eigenspectrum(X)
    #
    # # Plot representation of 2 dimensional tsne embedding
    # plot_tsne(X,labels)
    #
    #X = np.load("data/sentqs_skipgram_embedding.npy")
    #create_domain_adaptation_problem(X,tweets,labels,sentiment)

    #run_classification()


if __name__ == '__main__':

    # Obtain the all files of the dataset preprocessing, including plots, feature representation etc.
    # After running this file you will find the corresponding files for classification in the data folder
    main_preprocessing()
