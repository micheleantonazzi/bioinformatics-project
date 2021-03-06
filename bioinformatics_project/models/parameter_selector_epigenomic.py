import os
import pickle

import numpy
from tensorflow.keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback, TQDMCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored
from tensorflow.keras.callbacks import LearningRateScheduler

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *

"""
    This class store the parameters for each model in the ModelBuilderEpigenomic object.
    In particular, the Decision Tree and Random Forest parameters have been chosen using the Grid Search technique and,
    given its complexity, the selected parameters are saved to disk
    
"""
class ParameterSelectorEpigenomic:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_epigenomic_functions(self):
        return {
            DECISION_TREE_GRID: self.get_decision_tree_parameters_grid,
            RANDOM_FOREST_GRID: self.get_random_forest_parameters_grid,
            PERCEPTRON: self.get_perceptron_parameters,
            PERCEPTRON_2: self.get_perceptron_2_parameters,
            MLP: self.get_mlp_parameters,
            MLP_2: self.get_mlp_2_parameters,
            FFNN_1: self.get_ffnn_1_parameters,
            FFNN_2: self.get_ffnn_2_parameters,
            FFNN_3: self.get_ffnn_3_parameters,
            FFNN_4: self.get_ffnn_4_parameters,
            FFNN_5: self.get_ffnn_5_parameters,
            FFNN_6: self.get_ffnn_6_parameters,
            FFNN_7: self.get_ffnn_7_parameters
        }

    """
        This method load the model parameters from disk
    """
    def load_parameters_from_disk(self, model_type: str) -> dict:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters')

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, model_type + '.pkl')
        if os.path.exists(path):
            print(colored(f'Loading best parameters for {model_type} from file', 'green'))
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    """
        This method save the model parameters to disk
    """
    def save_parameters_to_disk(self, model_type: str, best_parameters):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters', model_type + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(best_parameters, f, pickle.HIGHEST_PROTOCOL)

    """
        Return Decision Tree best parameters found using the Grid Search technique.
        It the parameters have been already calculated, they are loaded from disk
    """
    def get_decision_tree_parameters_grid(self, _):
        best_parameters = self.load_parameters_from_disk(DECISION_TREE_GRID)

        if len(best_parameters.keys()) == 0:
            print(colored(f'Starting calculating best parameters for {DECISION_TREE_GRID}', 'red'))
            parameters = dict(
                max_depth=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, None],
                class_weight=[None, 'balanced'],
            )
            for region, (data, labels) in self._data.get_epigenomic_data_for_learning().items():
                grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters,
                                           n_jobs=-1, scoring='balanced_accuracy')
                grid_search.fit(data, labels)
                best_parameters[region] = {name: value for name, value in grid_search.best_params_.items()}
                best_parameters[region]['best_score'] = grid_search.best_score_

            self.save_parameters_to_disk(DECISION_TREE_GRID, best_parameters)

        for region, data in best_parameters.items():
            print(colored(f'Best {DECISION_TREE_GRID} parameters for {region}: ' + str(data), 'green'))
            data.pop('best_score', None)
        return best_parameters

    """
        Return the Random Forest best parameters found using the Grid Search technique.
        It the parameters have been already calculated, they are loaded from disk
    """
    def get_random_forest_parameters_grid(self, _=None):
        best_parameters = self.load_parameters_from_disk(RANDOM_FOREST_GRID)

        if len(best_parameters.keys()) == 0:
            print(colored(f'Starting calculating best parameters for {RANDOM_FOREST_GRID}', 'red'))
            parameters = dict(
                n_estimators=[60, 70, 80, 90, 100, 120, 140, 160],
                max_depth=[6, 8, 10, 12, 14, 16, 18, 20],
                class_weight=['balanced'],
            )
            for region, (data, labels) in self._data.get_epigenomic_data_for_learning().items():
                grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters,
                                           n_jobs=-1, scoring='balanced_accuracy')
                grid_search.fit(data, labels)
                best_parameters[region] = {name: value for name, value in grid_search.best_params_.items()}
                best_parameters[region]['best_score'] = grid_search.best_score_

            self.save_parameters_to_disk(RANDOM_FOREST_GRID, best_parameters)

        for region, data in best_parameters.items():
            print(colored(f'Best {RANDOM_FOREST_GRID} parameters for {region}: ' + str(data), 'green'))
            data.pop('best_score', None)
        return best_parameters

    def get_perceptron_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {PERCEPTRON} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_perceptron_2_parameters(self, _=None):
        parameters = dict(
             epochs=1000,
             batch_size=1024,
             validation_split=0.1,
             shuffle=True,
             verbose=False,
             callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {PERCEPTRON_2} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_mlp_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {MLP} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_mlp_2_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {MLP_2} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_1_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_1} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_2_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_2} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_3_parameters(self, labels):
        neg = numpy.count_nonzero(labels == 0)
        pos = numpy.count_nonzero(labels == 1)
        total = len(labels)
        class_weight = {0: (1 / neg) * (total) / 2.0, 1: (1 / pos) * (total) / 2.0}
        parameters_promoters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor='val_pr', mode="max", patience=50, restore_best_weights=True),
            ],
            class_weight=class_weight
        )

        neg = numpy.count_nonzero(labels == 0)
        pos = numpy.count_nonzero(labels == 1)
        total = len(labels)
        class_weight = {0: (1 / neg) * (total) / 2.0, 1: (1 / pos) * (total) / 2.0}
        parameters_enhancers = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor='val_pr', mode="max", patience=50, restore_best_weights=True),
            ],
            class_weight=class_weight
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters_promoters, DataRetrieval.KEY_ENHANCERS: parameters_enhancers}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_3} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_4_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=100,
            validation_split=0.1,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_4} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_5_parameters(self, _=None):
        parameters = dict(
            epochs=400,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_5} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_6_parameters(self, _=None):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_6} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_7_parameters(self, _):
        parameters = dict(
            epochs=200,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="auprc", mode="max", patience=10),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_7} parameters for {region}: ' + str(data), 'green'))
        return best_parameters


