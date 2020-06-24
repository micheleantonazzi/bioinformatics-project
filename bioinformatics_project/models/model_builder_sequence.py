from tensorflow.keras.metrics import AUC
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten

from bioinformatics_project.models.models_type import *
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class ModelBuilderSequence:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_sequence_functions(self):
        return {
            PERCEPTRON_SEQUENCE: self.create_perceptron,
            MLP_SEQUENCE: self.create_mlp
        }

    def create_perceptron(self):
        perceptron = Sequential([
            Input(shape=(200, 4)),
            Flatten(),
            Dense(1, activation="sigmoid")
        ], PERCEPTRON_SEQUENCE)

        perceptron.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
        )
        return perceptron

    def create_mlp(self):
        mlp = Sequential([
            Input(shape=(200, 4)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ], MLP_SEQUENCE)

        mlp.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
        )
        return mlp
