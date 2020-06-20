from keras_tqdm import TQDMNotebookCallback
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.callbacks import EarlyStopping

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *


class ModelBuilder:

    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_functions(self):
        return {
            DECISION_TREE_GRID: self.create_decision_tree,
            RANDOM_FOREST_GRID: self.create_random_forest,
            PERCEPTRON: self.create_perceptron,
            MLP: self.create_mlp
        }

    def create_decision_tree(self, _, parameters):
        return DecisionTreeClassifier(**parameters), {}

    def create_random_forest(self, _, parameters):
        return RandomForestClassifier(**parameters, n_jobs=-1), {}

    def create_perceptron(self, region, parameters):
        perceptron = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(1, activation="sigmoid")
        ], "Perceptron")

        perceptron.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )
        return perceptron, parameters

    def create_mlp(self, region, parameters):
        mlp = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ], "MLP")

        mlp.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )
        return mlp, parameters
