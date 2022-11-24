from os.path import exists
from urllib.request import urlopen
import torch

import numpy as np

def download_data():
    if not exists("data/eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())

def eeg_data():
    download_data()
    data = np.loadtxt("data/eeg.dat", delimiter=",", skiprows=19)
    print("[raw data shape] {}".format(data.shape))
    data = torch.tensor(data[::20, :-1]).double()
    return data