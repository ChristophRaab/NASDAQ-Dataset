"""
NSDQS - NASDAQ Stream Dataset

Contens of this file:
Fetch from twitter
Create sentiment label

authors:  Christoph Raab
"""

import twint
import datetime
from textblob import TextBlob
# Function to get sentiment
import pandas as pd
import os
import datetime

def sentiment_anylsis(sentence):
    temp = TextBlob(sentence).sentiment[0]
    if temp == 0.0:
        return 0.0 # Neutral
    elif temp >= 0.0:
        return 1.0 # Positive
    else:
        return 2.0 # Negative

def fetch_tweets(hastag,date,limit=1):
    c = twint.Config()
    c.Search = "#"+ hastag
    c.Store_object = True
    c.Since = date
    c.Lang = "en"
    c.Limit = limit
    twint.run.Search(c)
    tweets = twint.output.tweets_list
    return [t.tweet for t in tweets]

def create_dates(numdays):
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]
    return date_list

def main_fetch():
    hashtags = ['ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG', 'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX',
     'TSLA', 'XEL']
    dates = create_dates(300)
    date = dates[-1].strftime("%Y-%m-%d")
    for hashtag in hashtags:
        tweets = fetch_tweets(hashtag,date,limit=10000)
    tweets = list(set(tweets))
    sentiments = [ [t,sentiment_anylsis(t)] for t in tweets ]
    df = pd.DataFrame(sentiments,columns=["Tweets","Sentiment"])
    df.to_csv("Tweets"+datetime.date.today().strftime("%d_%m_%y")+".csv")

if __name__ == '__main__':
    main_fetch()