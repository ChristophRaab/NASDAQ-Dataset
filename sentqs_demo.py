"""
NSDQS - NASDAQ Domain Adaptation Dataset

Contens of this file:
Domain Adaptation Demo
authors:  Christoph Raab
"""

# Error: default_graph when using only keras and not tensorflow.keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Conv2D, BatchNormalization
from tensorflow.keras.datasets import imdb
import requests
import sys
import numpy as np
from NBT import NBT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
link = "https://cloud.fhws.de/index.php/s/4sJ69ocZW8epAke/download"

file_name = "data/sentqs_dataset.npz"

def download_data():
    with open(file_name, "wb") as f:
            print("Downloading %s" % file_name)
            response = requests.get(link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None: # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                    sys.stdout.flush()

    data = np.load(file_name,allow_pickle=True)
    Xs = data["arr_0"]
    Ys = data["arr_1"]
    Xt = data["arr_2"]
    Yt = data["arr_3"]
    return Xs,Ys,Xt,Yt

# Xs,Ys,Xt,Yt = download_data()

#If dataset file is already downloaded
data = np.load(file_name,allow_pickle=True)
Xs = data["arr_0"]
Ys = data["arr_1"]
Xt = data["arr_2"]
Yt = data["arr_3"]

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# Training
batch_size = 256
epochs = 10
# LSTM

lstm_output_size = 70
Xs = np.expand_dims(Xs, 2)
Xt = np.expand_dims(Xt, 2)


model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(35))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()
print('Train...')
model.fit(Xs, Ys,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(Xt, Yt))
score, acc = model.evaluate(Xt, Yt, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)