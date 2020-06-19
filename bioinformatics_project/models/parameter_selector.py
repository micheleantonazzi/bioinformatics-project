import os
import pickle

from tensorflow.keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.model_builder import ModelBuilder
from bioinformatics_project.models.models_type import *


class ParameterSelector:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_functions(self):
        return {
            DECISION_TREE: self.get_decision_tree_parameters,
            RANDOM_FOREST: self.get_random_forest_parameters
        }

    def load_parameters_from_disk(self, model_type: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters')

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, model_type + '.pkl')
        if os.path.exists(path):
            print(colored(f'Loading best parameters for {model_type} from file', 'green'))
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    def save_parameters_to_disk(self, model_type: str, best_parameters):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters', model_type + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(best_parameters, f, pickle.HIGHEST_PROTOCOL)

    def get_decision_tree_parameters(self):
        best_parameters = self.load_parameters_from_disk(DECISION_TREE)

        if best_parameters is None:
            print(colored(f'Starting calculating best parameters for {DECISION_TREE}', 'red'))
            parameters = dict(
                max_depth=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, None],
                class_weight=[None, 'balanced'],
            )
            for region, (data, labels) in self._data.get_epigenomic_data_for_learning().items():
                grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters, n_jobs=-1)
                grid_search.fit(data, labels)
                best_parameters[region] = {name: value for name, value in grid_search.best_params_.items()}
                best_parameters[region]['best_score'] = grid_search.best_score_

            self.save_parameters_to_disk(DECISION_TREE, best_parameters)

        for region, data in best_parameters.items():
            print(colored(f'Best {DECISION_TREE} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_random_forest_parameters(self):
        best_parameters = self.load_parameters_from_disk(RANDOM_FOREST)

        if best_parameters is None:
            print(colored(f'Starting calculating best parameters for {RANDOM_FOREST}', 'red'))
            parameters = dict(
                n_estimators=[10, 20, 50, 70, 100, 200],
                max_depth=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, None],
                class_weight=['balanced', None],
            )
            for region, (data, labels) in self._data.get_epigenomic_data_for_learning().items():
                grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, n_jobs=-1)
                grid_search.fit(data, labels)
                best_parameters[region] = {name: value for name, value in grid_search.best_params_.items()}
                best_parameters[region]['best_score'] = grid_search.best_score_

            self.save_parameters_to_disk(RANDOM_FOREST, best_parameters)

        for region, data in best_parameters.items():
            print(colored(f'Best {RANDOM_FOREST} parameters for {region}: ' + str(data), 'green'))
        return best_parameters


