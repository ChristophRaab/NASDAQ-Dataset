"""
NSDQS - NASDAQ Stream Dataset

Contens of this file:
Demo stream classification

authors:  Christoph Raab
"""

import requests
import numpy as np
import sys
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.data_stream import DataStream
from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTree
from skmultiflow.lazy.sam_knn import SAMKNN
link = "https://cloud.fhws.de/index.php/s/r4R8S3Tc6j3xK6T/download"
file_name = "data/nsdqs_stream_skipgram.npy"

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

    return data

# data = download_data()
#If dataset file is already downloaded
data = np.load(file_name,allow_pickle=True)

sam = SAMKNN()
arf = ARFHoeffdingTree()

stream = DataStream(data[:,1:],data[:,0].astype(int))
stream.prepare_for_use()

evaluator = EvaluatePrequential(max_samples=10000,
                                max_time=1000,
                                show_plot=True,
                                metrics=['accuracy', 'kappa'])

evaluator.evaluate(stream=stream, model=[sam,arf], model_names=['Sam','RSLVQ'])


