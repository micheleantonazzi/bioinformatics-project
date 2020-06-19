import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *


class ParameterSelector:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_best_parameters(self, model_type: str):
        best_parameters = {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters')

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, model_type + '.pkl')
        if os.path.exists(path):
            print(colored(f'Loading best parameters for {model_type} from file', 'green'))
            with open(path, 'rb') as f:
                best_parameters = pickle.load(f)
        else:
            print(colored(f'Starting calculating best parameters for {model_type}', 'red'))

            parameters = dict()
            classifier = None
            if model_type == DECISION_TREE:
                parameters = dict(
                    max_depth=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, None],
                    class_weight=[None, 'balanced'],
                )
                classifier = DecisionTreeClassifier()

            elif model_type == RANDOM_FOREST:
                parameters = dict(
                    n_estimators=[10, 20, 50, 70, 100, 200],
                    max_depth=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, None],
                    class_weight=['balanced', None],
                )
                classifier = RandomForestClassifier()

            for region, (data, labels) in self._data.get_epigenomic_data_for_learning().items():
                random_search = GridSearchCV(classifier, parameters, scoring='balanced_accuracy', n_jobs=-1)
                random_search.fit(data, labels)
                best_parameters[region] = {name: value for name, value in random_search.best_params_.items()}
                best_parameters[region]['best_score'] = random_search.best_score_

            with open(path, 'wb') as f:
                pickle.dump(best_parameters, f, pickle.HIGHEST_PROTOCOL)

        for region, data in best_parameters.items():
            print(colored(f'Best {model_type} parameters for {region}: ' + str(data), 'green'))
        return best_parameters
