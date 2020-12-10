# NASDAQ Twitter Feed Dataset
Streaming and domain adaptation datasets based on the twitter feed 
with hashtags related to NASDAQ and represents new challenges in the domains.

The dataset contains tweets from twitter crawled from 10.02.2019 til 3.12.2019.
The tweets from streaming and domain adaptation are chosen with respect to the tasks and can be found below.
All tweets are crawled that no user information is passed to us and only the tweet itself is processed. 

This repository offers two datasets.
* The prefix `nsdqs_` includes files for stream dataset.
* The prefix `sentqs_` includes files for domain adaptation dataset.

## NSDQ Dataset for Stream Analysis 
* The main dataset file can be found in `data/nsdqs_skipgram_embedding.npy`. 
* Hastags crawled: ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG',
'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX', 'TSLA' and 'XEL'.
* The dataset __30278 tweets__ with __1000 feature dimensions__.
* Number of classes: 15

### Scenario
__Test-Then-Train__<br/>
A primary challenge in the analysis and monitored classification of data streams in real-time is the changing underlying concept. This is called concept drift. This forces the machine learning algorithms to adapt constantly. This data set consists of tweets of the NASDAQ codes of the largest American companies and reflects the volatility of the stock market. Due to this volatility, many different concept drifts exist and pose a new challenge in the stream context, as there is no underlying systematic that explains or makes the drift predictable. The data set is highly unbalanced and very high-dimensional compared to other stream data sets. 

### Challanges
* High feature dimension compared to existing dataset.
* High number of classes with large imbalances compared to existing dataset.
* High volatile dataset with many non-specified concept drifts.

### Usage
* (Optional) Preprocess on your on:
    1. Raw tweets are at `Tweets.csv`
    2. Run `nsdqs_processing.py`
    3. This creates a basic statistical dataset description, trains the embedding and plots tsne embedding and eigenspectra which needs some time. 
* Store dataset ready for usage in `data/nsdqs_stream_skipgram.npy`.
  
* Demo 
Run `nsqds_demo.py` for a stream machine learning demonstration using SamKNN and RSVLQ. 
  
## SentQS Dataset for Domain Adaptation 
* The main dataset file can be found in `data/sentqs_skipgram_embedding.npy`. 
* Hastags crawled: 'ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG',
'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX', 'TSLA', 'XEL', __'positive', 'bad' and 'sad'__.
* The dataset __61536 tweets__ with __300 feature dimensions__.
* Number of classes: 3 (Positive, Neutral, Negative Sentiment)

### Scenario
__Train on Sentiment Tweets - Evaluate Sentiment of Coperate Tweets__<br/>
__Change of Language Distribution between Train and Test dataset__<br/>
If the scenario of different distributions between the training and the test data set is encountered, it is called a Domain Adaptation Problem. In contrast to other Domain Adaptation Data Sets, which are mostly image data sets or which are not subject to a real scenario, this data set offers a transfer learning scenario in the context of Social Media Analysis. 
The core idea is to learn a sentiment analysis for positive, neutral and negative tweets. Moreover, to apply this through domain adaptation to corporate tweets to unseen coperations. The practical advantage is that there is no need for manual labeling of the company tweets and they cover a large language spectrum. 


### Challanges
* Real-world scenario not relying on standard image or text dataset with exhausting preprocessing. 
* High number of samples compared to existing datasets.
* Highly unbalanced Classes.
* Domain adaptation problem implicity by using tweets from varying hashtags.

### Usage
* (Optional) Preprocess on your on:
    1. Raw tweets are at `Tweets.csv`
    2. Run `sentqs_processing.py`
    3. This creates a basic statistical dataset description, trains the embedding and plots tsne embedding and eigenspectra which needs some time. 
* Store dataset ready for usage in `data/sentqs_da_skigram.npy`.
  
* Demo 
Run `sentqs_demo.py` for a stream machine learning demonstration using SamKNN and RSVLQ. 
