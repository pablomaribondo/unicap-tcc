import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV

from helpers import fft, read, save
from constants import DATA_PATH, TRAINING_MATRIX, TRAINING_LABELS, TEST_MATRIX, TEST_LABELS, OUTPUT_FILES


def process():
    PRECISION = 1024
    TRAINING_RATIO = 8
    TEST_RATIO = 2

    folders = list(filter(lambda name: len(
        name.split('.')) == 1, os.listdir(DATA_PATH)))
    sorted_folders = sorted(folders, key=lambda folder: float(folder[1:]))

    data = {
        TRAINING_MATRIX: [],
        TEST_MATRIX: [],
        TRAINING_LABELS: [],
        TEST_LABELS: []
    }

    for current_label, folder in enumerate(sorted_folders):
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

            data[TRAINING_MATRIX].extend(
                list(map(lambda chunk: fft(chunk, PRECISION), chunks[:TRAINING_RATIO])))
            data[TEST_MATRIX].extend(
                list(map(lambda chunk: fft(chunk, PRECISION), chunks[-TEST_RATIO:])))
            data[TRAINING_LABELS].extend([current_label] * TRAINING_RATIO)
            data[TEST_LABELS].extend([current_label] * TEST_RATIO)
    return data


def fetch_data(output_files):
    data = {}

    if (OUTPUT_FILES[TRAINING_MATRIX] and
        OUTPUT_FILES[TEST_MATRIX] and
        OUTPUT_FILES[TRAINING_LABELS] and
            OUTPUT_FILES[TEST_LABELS] in output_files):
        data[TRAINING_MATRIX] = read(OUTPUT_FILES[TRAINING_MATRIX])
        data[TEST_MATRIX] = read(OUTPUT_FILES[TEST_MATRIX])
        data[TRAINING_LABELS] = read(OUTPUT_FILES[TRAINING_LABELS])
        data[TEST_LABELS] = read(OUTPUT_FILES[TEST_LABELS])
    else:
        data = process()

        save(data[TRAINING_MATRIX], OUTPUT_FILES[TRAINING_MATRIX])
        save(data[TEST_MATRIX], OUTPUT_FILES[TEST_MATRIX])
        save(data[TRAINING_LABELS], OUTPUT_FILES[TRAINING_LABELS])
        save(data[TEST_LABELS], OUTPUT_FILES[TEST_LABELS])
    return data


def train(model_name, model_method, params_grid, output_files, data, iterations):
    models = []

    if (OUTPUT_FILES[model_name] in output_files):
        models = read(OUTPUT_FILES[model_name])
    else:
        model = model_method

        for _ in range(iterations):
            print(f'{model_name}: {_}')
            cv_model = GridSearchCV(estimator=model, param_grid=params_grid)
            cv_model.fit(data[TRAINING_MATRIX], data[TRAINING_LABELS])
            models.append(cv_model)
        save(models, OUTPUT_FILES[model_name])
    return models
