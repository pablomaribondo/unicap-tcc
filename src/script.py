import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV

from helpers import fft, read, save, output_filename
from constants import DATA_PATH, TEST_RATIO, TRAINING_RATIO


def train_test_split():
    PRECISION = 1024

    folders = list(filter(lambda name: len(
        name.split('.')) == 1, os.listdir(DATA_PATH)))
    sorted_folders = sorted(folders, key=lambda folder: float(folder[1:]))

    data = {
        'X_train': [],
        'y_train': [],
        'X_test': [],
        'y_test': []
    }

    for index, folder in enumerate(sorted_folders):
        files = os.listdir(f'{DATA_PATH}/{folder}')

        for file_name in files:
            file = loadmat(f'{DATA_PATH}/{folder}/{file_name}')
            raw_data = file['data'][0][0][0]
            first_sensor = raw_data[0]

            min_value = np.min(first_sensor)
            max_value = np.max(first_sensor)
            normalized_data = (first_sensor - min_value) / \
                (max_value - min_value)

            chunks = np.array_split(
                normalized_data, TRAINING_RATIO + TEST_RATIO)

            data['X_train'].extend(
                list(map(lambda chunk: fft(chunk, PRECISION), chunks[:TRAINING_RATIO])))
            data['y_train'].extend([index] * TRAINING_RATIO)
            data['X_test'].extend(
                list(map(lambda chunk: fft(chunk, PRECISION), chunks[-TEST_RATIO:])))
            data['y_test'].extend([index] * TEST_RATIO)
    array_data = {
        'training_matrix': np.array(data['X_train']),
        'training_labels': np.array(data['y_train']),
        'test_matrix': np.array(data['X_test']),
        'test_labels': np.array(data['y_test'])
    }

    return array_data


def fetch_data(file_name, output_files):
    train_test_file = output_filename(file_name)

    if (train_test_file in output_files):
        return read(train_test_file)

    data = train_test_split()
    save(data, train_test_file)

    return data


def fit(model_name, model_method, params_grid, output_files, data, iterations):
    model_file = output_filename(model_name)
    X, y = data

    if (model_file in output_files):
        return read(model_file)

    models = []
    for index in range(iterations):
        print(f'Training {model_name} n.{index}')
        cv_model = GridSearchCV(estimator=model_method, param_grid=params_grid)
        cv_model.fit(X, y)
        models.append(cv_model)

    save(models, model_file)

    return models


def score(file_name, models, output_files, data):
    score_file = output_filename(file_name)
    X, y = data

    if (score_file in output_files):
        return read(score_file)

    scores = []
    for index, model in enumerate(models):
        print(f'Scoring {file_name} n.{index}')
        score_data = model.score(X, y)
        scores.append(score_data)

    save(scores, score_file)

    return scores
