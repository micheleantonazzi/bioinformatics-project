from sklearn.tree import DecisionTreeClassifier

from bioinformatics_project.models.models_type import *


class ModelBuilder:

    def __init__(self):
        pass

    def create_model(self, type: str, parameters):
        if type == DECISION_TREE:
            return DecisionTreeClassifier(
                **parameters
            )
