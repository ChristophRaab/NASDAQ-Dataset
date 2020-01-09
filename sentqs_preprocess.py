
from datetime import timedelta, date
import pandas as pd
import numpy
from analysis.analyse_sentiment_conv import train_model_convolutional
from keras.engine.saving import load_model
from learn.learn_embeddings import learn_embeddings
from analysis.analyse_sentiment_conv import predict_model_convolutional
from base.utils import load_model_mat
from fetch.download_tweets import download_tweets_twint
from fetch.fetch_config import FetchConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.preprocess import preprocess, tokenize_cleaned_tweets
from nltk.corpus import stopwords
import re
df = pd.read_csv("Tweets.csv")
hashtags = ['ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG', 'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX',
            'TSLA', 'XEL', 'positive', 'bad', 'sad']

d = {k: [] for k in hashtags}
for t in df.iloc[:, 1]:
    for h in hashtags:
        if "#" + h.lower() in t.lower():
            d[h].append(t)
i = 0
for l in d.values():
    i = i + len(l)
print(i == len(df))

cleaned_tweets = preprocess(d)

# step 3: tokenization (using the tokenizer created in the fetch_build_model.py script)
tokenized_tweets = tokenize_cleaned_tweets(cleaned_tweets)
encoded_categories = {hashtags[i]: i for i in range(len(hashtags))}
y = []  # eg. 0,0,0,1,1,1,1
for hashtag, tweets in tokenized_tweets.items():
    for item in tweets:
        y.append(encoded_categories[hashtag])
x = numpy.concatenate(list(tokenized_tweets.values()))


# step 4: creating embeddings
glove, word, sg = learn_embeddings(x, y, cleaned_tweets)

# step 5: learn model
train_model_convolutional(x, y, [word, glove, sg])

# step 5: learn model
model = load_model('sentiment_conv_ep100.h5')
predictions = predict_model_convolutional(x, model=model, evaluate=True, y=y)
print(predictions[0])