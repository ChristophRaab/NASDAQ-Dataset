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
The main dataset file can be found in `data/nsdqs_skipgram_embedding.npy`. 
Hastags crawled: ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG',
'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX', 'TSLA' and 'XEL'.
There are number of tweets with 1000 data dimensions.
Number of classes: 17

### Scenario
A primary challenge in the analysis and monitored classification of data streams in real-time is the changing underlying concept. This is called concept drift. This forces the machine learning algorithms to adapt constantly. This data set consists of tweets of the NASDAQ codes of the largest American companies and reflects the volatility of the stock market. Due to this volatility, many different concept drifts exist and pose a new challenge in the stream context, as there is no underlying systematic that explains or makes the drift predictable. The data set is highly unbalanced and very high-dimensional compared to other stream data sets. 

### Challanges
* High feature dimension compared compared to existing dataset.
* High number of classes with large imbalances compared to existing dataset.
* High volatile dataset with many non-specified datasets.

### Usage
1. Download data from: 
2. Copy it to data/ 
3. (Optional) Preprocess on your on:
  - Raw tweets are at `Tweets.csv`
  - Run `nsdqs_processing.py`
  - This creates a basic statistical dataset description, trains the embedding and plots tsne embedding and eigenspectra which needs some time. 
4. Obtain dataset ready for usage in `data/nsdqs_skipgram_embedding.npy`.
  
* Demo 
Run `nsqds_demo.py` for a stream machine learning demonstration using SamKNN and RSVLQ. 
  
## SentQS Dataset for Domain Adaptation 
The main dataset file can be found in `data/sentqs_skipgram_embedding.npy`. 
Hastags crawled: 'ADBE', 'GOOGL', 'AMZN', 'AAPL', 'ADSK', 'BKNG',
'EXPE', 'INTC', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 'SBUX', 'TSLA', 'XEL', __'positive', 'bad' and 'sad'__.
There are tweets encoded with 200 data dimensions.

### Scenario

### Challanges
* Real-world scenario not relying on standard image or text datafield undergoing large preprossing. 
* High number of samples compared to existing datasets.
* Highly unbalanced Classes.
* Domain adaptation problem implicity by using tweets from varying hashtags.

### Usage
1. Download data from: 
2. Copy it to data/ 
3. (Optional) Preprocess on your on:
  - Raw tweets are at `Tweets.csv`
  - Run `sentqs_processing.py`
  - This creates a basic statistical dataset description, trains the embedding and plots tsne embedding and eigenspectra which needs some time. 
4. Obtain dataset ready for usage in `data/sentqs_skipgram_embedding.npy` for usage.
  
* Demo 
Run `sentqs_demo.py` for a stream machine learning demonstration using SamKNN and RSVLQ. 
