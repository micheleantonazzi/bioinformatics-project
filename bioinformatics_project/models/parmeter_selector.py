import os
import pickle

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *


class ParameterSelector:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_best_parameters(self, model_type: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters')
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, DECISION_TREE + '.pkl')
        if os.path.exists(path):
            print(colored(f'Loading best parameters for {model_type} from file', 'green'))
            with open(path, 'rb') as f:
                return pickle.load(f)

        best_parameters = {region: dict() for region in [DataRetrieval.KEY_PROMOTERS, DataRetrieval.KEY_ENHANCERS]}

        print(colored(f'Starting calculating best parameters for {model_type}', 'red'))
        if model_type == DECISION_TREE:
            parameters = dict(
                max_depth=[2, 5, 10, 20, 30, 50, 100, 150, 200, 300, 400, None],
                class_weight=[None, "balanced"]
            )
            for region, (data, labels) in self._data.get_epigenomic_data_for_learning().items():
                random_search = GridSearchCV(DecisionTreeClassifier(), parameters, scoring="balanced_accuracy", n_jobs=8)
                random_search.fit(data, labels)
                print(f'For {region} the best parameters for {model_type} are {random_search.best_params_} with accurancy of {random_search.best_score_}')
                best_parameters[region] = {name: value for name, value in random_search.best_params_.items()}
                best_parameters[region]['best_score'] = random_search.best_score_

        with open(path, 'wb') as f:
            pickle.dump(best_parameters, f, pickle.HIGHEST_PROTOCOL)
