from tensorflow.keras.metrics import AUC
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D

from bioinformatics_project.models.models_type import *
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class ModelBuilderSequence:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_sequence_functions(self):
        return {
            PERCEPTRON_SEQUENCE: self.create_perceptron,
            MLP_SEQUENCE: self.create_mlp,
            FFNN_SEQUENCE: self.create_ffnn,
            CNN: self.create_cnn
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

    def create_ffnn(self):
        ffnn = Sequential([
            Input(shape=(200, 4)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ], FFNN_SEQUENCE)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
        )

        return ffnn

    def create_cnn(self):
        cnn = Sequential([
            Input(shape=(200, 4)),
            Reshape((200, 4, 1)),
            Conv2D(64, kernel_size=(10, 2), activation="relu"),
            Conv2D(64, kernel_size=(10, 2), activation="relu"),
            Dropout(0.3),
            Conv2D(32, kernel_size=(10, 2), strides=(2, 1), activation="relu"),
            Conv2D(32, kernel_size=(10, 1), activation="relu"),
            Conv2D(32, kernel_size=(10, 1), activation="relu"),
            Dropout(0.3),
            Flatten(),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ], CNN)

        cnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
        )

        return cnn