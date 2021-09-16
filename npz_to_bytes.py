import numpy as np
import tensorflow as tf

DATA = r"C:\Users\Chris\Workspaces\NASDAQ-Dataset\data\sentqs_skipgram_sentence_embedding.npz"

with np.load(DATA) as data:
    tweets = data['embedding']
    tweets.dtype = np.float32
    tweets.tofile(r'C:\Users\Chris\Workspaces\NASDAQ-Dataset\data\sentqs_skipgram_sentence_embedding.bytes')
