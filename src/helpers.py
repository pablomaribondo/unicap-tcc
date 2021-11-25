import numpy as np
import pickle

from constants import OUTPUT_PATH


def fft(data, precision):
    return np.log2(np.abs(np.fft.fft(data, n=precision)) + 0.001)


def save(data, file_name):
    with open(f'{OUTPUT_PATH}/{file_name}', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def read(file_name):
    with open(f'{OUTPUT_PATH}/{file_name}', 'rb') as f:
        return pickle.load(f)
