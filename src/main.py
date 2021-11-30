import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


from helpers import best_estimators, output_filename, read, save
from script import fetch_data, fit, score
from constants import (OUTPUT_PATH, PARAMS_GRID, TEST_RATIO, TRAINING_RATIO)

output_files = list(filter(lambda name: name.split(
    '.')[-1] == 'pickle', os.listdir(OUTPUT_PATH)))
data = fetch_data('train_test', output_files)

training_matrix = data['training_matrix']
training_labels = data['training_labels']
test_matrix = data['test_matrix']
test_labels = data['test_labels']

ITERATIONS = 20

knn_models = fit('knn_models', KNeighborsClassifier(
), PARAMS_GRID['KNN'], output_files, data=(training_matrix, training_labels), iterations=ITERATIONS)
bag_models = fit('bag_models', BaggingClassifier(DecisionTreeClassifier(
)), PARAMS_GRID['BAG'], output_files, data=(training_matrix, training_labels), iterations=ITERATIONS)
rf_models = fit('rf_models', RandomForestClassifier(
), PARAMS_GRID['RF'], output_files, data=(training_matrix, training_labels), iterations=ITERATIONS)
et_models = fit('et_models', ExtraTreesClassifier(
), PARAMS_GRID['ET'], output_files, data=(training_matrix, training_labels), iterations=ITERATIONS)
mlp_models = fit('mlp_models', MLPClassifier(
), PARAMS_GRID['MLP'], output_files, data=(training_matrix, training_labels), iterations=ITERATIONS)
svc_models = fit('svc_models', SVC(), PARAMS_GRID['SVC'], output_files, data=(
    training_matrix, training_labels), iterations=ITERATIONS)

voting_models = []
voting_file = output_filename('voting_models')
if (voting_file in output_files):
    voting_models = read(voting_file)
else:
    voting_estimators = []
    for index in range(ITERATIONS):
        estimators = [
            ('KNN', knn_models[index].best_estimator_),
            ('BAG', bag_models[index].best_estimator_),
            ('RF', rf_models[index].best_estimator_),
            ('ET', et_models[index].best_estimator_),
            ('MLP', mlp_models[index].best_estimator_),
            ('SVC', svc_models[index].best_estimator_)
        ]
        voting_estimators.append(estimators)

    for index, estimator in enumerate(voting_estimators):
        print(f'Training voting_models n.{index}')
        model = VotingClassifier(estimators=estimator, voting='hard')

        model.fit(training_matrix, training_labels)
        voting_models.append(model)
    save(voting_models, voting_file)

knn_scores = score('knn_scores', best_estimators(knn_models),
                   output_files, data=(test_matrix, test_labels))
bag_scores = score('bag_scores', best_estimators(bag_models),
                   output_files, data=(test_matrix, test_labels))
rf_scores = score('rf_scores', best_estimators(rf_models),
                  output_files, data=(test_matrix, test_labels))
et_scores = score('et_scores', best_estimators(et_models),
                  output_files, data=(test_matrix, test_labels))
mlp_scores = score('mlp_scores', best_estimators(mlp_models),
                   output_files, data=(test_matrix, test_labels))
svc_scores = score('svc_scores', best_estimators(svc_models),
                   output_files, data=(test_matrix, test_labels))
voting_scores = score('voting_scores', voting_models,
                      output_files, data=(test_matrix, test_labels))


names = ['KNN', 'BAG', 'RF', 'ET', 'MLP', 'SVC', 'VTN']
scores = [knn_scores, bag_scores, rf_scores,
          et_scores, mlp_scores, svc_scores, voting_scores]

num_classes = TRAINING_RATIO + TEST_RATIO

keras_model = Sequential([
    Input(shape=training_matrix.shape[1:]),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


keras_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam', metrics=['sparse_categorical_accuracy'])

keras_model.summary()

history = keras_model.fit(training_matrix, training_labels, validation_data=(
    test_matrix, test_labels), epochs=100)

test_loss, test_accuracy = keras_model.evaluate(
    test_matrix, test_labels, verbose=2)

# fig, ax = plt.subplots()
# ax.set_title('Algorithm Comparison')
# ax.boxplot(scores)
# ax.set_xticklabels(names)
# plt.show()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
