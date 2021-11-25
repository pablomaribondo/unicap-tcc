import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from helpers import read, save
from script import fetch_data, train
from constants import (OUTPUT_EXTENSION, OUTPUT_FILES, OUTPUT_PATH, KNN_MODELS, BAG_MODELS, RF_MODELS, ET_MODELS,
                       MLP_MODELS, SVC_MODELS, TEST_LABELS, TEST_MATRIX, TRAINING_LABELS, TRAINING_MATRIX, VOTING_MODEL)

output_files = list(filter(lambda name: name.split(
    '.')[-1] == OUTPUT_EXTENSION, os.listdir(OUTPUT_PATH)))
data = fetch_data(output_files)

ITERATIONS = 10

knn_params_grid = {
    'n_neighbors': [10, 50, 100, 200, 500],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}
knn_models = train(KNN_MODELS, KNeighborsClassifier(),
                   knn_params_grid, output_files, data, ITERATIONS)

bag_params_grid = {
    'n_estimators': [10, 50, 100, 200, 500]
}
bag_models = train(BAG_MODELS, BaggingClassifier(
    DecisionTreeClassifier()), bag_params_grid, output_files, data, ITERATIONS)

rf_params_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_models = train(RF_MODELS, RandomForestClassifier(),
                  rf_params_grid, output_files, data, ITERATIONS)

et_params_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2']
}
et_models = train(ET_MODELS, ExtraTreesClassifier(),
                  et_params_grid, output_files, data, ITERATIONS)

mlp_params_grid = {
    'max_iter': [1, 5, 50, 100, 200],
    'hidden_layer_sizes': [(10,), (20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}
mlp_models = train(MLP_MODELS, MLPClassifier(),
                   mlp_params_grid, output_files, data, ITERATIONS)

svc_params_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
svc_models = train(SVC_MODELS, SVC(),
                   svc_params_grid, output_files, data, ITERATIONS)

print(
    f'K-Nearest Neighbors (k-NN): {max(map(lambda knn_model: knn_model.best_score_, knn_models)) * 100}%')
print(
    f'Bagged Decision Trees: {max(map(lambda bag_model: bag_model.best_score_, bag_models)) * 100}%')
print(
    f'Random Forest: {max(map(lambda rf_model: rf_model.best_score_, rf_models)) * 100}%')
print(
    f'Extra Trees: {max(map(lambda et_model: et_model.best_score_, et_models)) * 100}%')
print(
    f'Multi-layer Perceptron: {max(map(lambda mlp_model: mlp_model.best_score_, mlp_models)) * 100}%')
print(
    f'C-Support Vector Classification: {max(map(lambda svc_model: svc_model.best_score_, svc_models)) * 100}%')


estimators = []
for index in range(ITERATIONS):
    estimators.append((f'knn_{index + 1}', knn_models[index].best_estimator_))
    estimators.append((f'bag_{index + 1}', bag_models[index].best_estimator_))
    estimators.append((f'rf_{index + 1}', rf_models[index].best_estimator_))
    estimators.append((f'et_{index + 1}', et_models[index].best_estimator_))
    estimators.append((f'mlp_{index + 1}', mlp_models[index].best_estimator_))
    estimators.append((f'svc_{index + 1}', svc_models[index].best_estimator_))


ensemble = ''
if (OUTPUT_FILES[VOTING_MODEL] in output_files):
    ensemble = read(OUTPUT_FILES[VOTING_MODEL])
else:
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(data[TRAINING_MATRIX], data[TRAINING_LABELS])
    save(ensemble, OUTPUT_FILES[VOTING_MODEL])

print(f'Voting: {ensemble.score(data[TEST_MATRIX], data[TEST_LABELS]) * 100}%')

# criar um box plot do resultado
# Rede neural para contraprova -> keras python
