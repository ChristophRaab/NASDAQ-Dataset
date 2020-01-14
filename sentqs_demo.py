"""
NSDQS - NASDAQ Domain Adaptation Dataset

Contens of this file:
Domain Adaptation Demo
authors:  Christoph Raab
"""


import requests
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

link = "https://cloud.fhws.de/index.php/s/M4rkbHj9FfW6YKo/download"
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

Xs,Ys,Xt,Yt = download_data()
#If dataset file is already downloaded
data = np.load(file_name,allow_pickle=True)
Xs = data["arr_0"]
Ys = data["arr_1"]
Xt = data["arr_2"]
Yt = data["arr_3"]

print("\n Classification Task Test \n")
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(Xs, Ys)
print(clf.score(Xt, Yt))