import os
import pickle

from tensorflow.keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tensorflow_addons.callbacks import TQDMProgressBar
from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *


class ParameterSelector:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_functions(self):
        return {
            DECISION_TREE_GRID: self.get_decision_tree_parameters_grid,
            RANDOM_FOREST_GRID: self.get_random_forest_parameters_grid,
            PERCEPTRON: self.get_perceptron_parameters,
            MLP: self.get_mlp_parameters
        }

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

    def save_parameters_to_disk(self, model_type: str, best_parameters):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters', model_type + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(best_parameters, f, pickle.HIGHEST_PROTOCOL)

    def get_decision_tree_parameters_grid(self):
        best_parameters = self.load_parameters_from_disk(DECISION_TREE_GRID)

        if len(best_parameters.keys()) == 0:
            print(colored(f'Starting calculating best parameters for {DECISION_TREE_GRID}', 'red'))
            parameters = dict(
                max_depth=[5, 10, 20, 35, 50, 65, 80, 100, None],
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

    def get_random_forest_parameters_grid(self):
        best_parameters = self.load_parameters_from_disk(RANDOM_FOREST_GRID)

        if len(best_parameters.keys()) == 0:
            print(colored(f'Starting calculating best parameters for {RANDOM_FOREST_GRID}', 'red'))
            parameters = dict(
                n_estimators=[50, 100, 150, 200, 300, 400, 500],
                max_depth=[5, 10, 20, 35, 50, 65, 80, 100, None],
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

    def get_perceptron_parameters(self):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50)]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {PERCEPTRON} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_mlp_parameters(self):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50)]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {MLP} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

