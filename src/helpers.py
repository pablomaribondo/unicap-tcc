import numpy as np
import pickle

from constants import OUTPUT_PATH


def fft(data, precision):
    return np.log2(np.abs(np.fft.fft(data, n=precision)) + 0.001)


def save(data, file_name, output_path=OUTPUT_PATH):
    with open(f'{output_path}/{file_name}', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def read(file_name, output_path=OUTPUT_PATH):
    with open(f'{output_path}/{file_name}', 'rb') as f:
        return pickle.load(f)


def best_estimators(models):
    return list(map(lambda model: model.best_estimator_, models))


def output_filename(file_name):
    return f'{file_name}.pickle'
