BASE_PATH = './src'
DATA_PATH = f'{BASE_PATH}/data'
OUTPUT_PATH = f'{BASE_PATH}/output'

TRAINING_RATIO = 8
TEST_RATIO = 2

PARAMS_GRID = {
    'KNN':  {
        'n_neighbors': [10, 50, 100, 200, 500],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
    },
    'BAG': {
        'n_estimators': [10, 50, 100, 200, 500]
    },
    'RF': {
        'n_estimators': [10, 50, 100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'ET': {
        'n_estimators': [10, 50, 100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'MLP': {
        'max_iter': [1, 5, 50, 100, 200],
        'hidden_layer_sizes': [(10,), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    },
    'SVC': {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }
}
