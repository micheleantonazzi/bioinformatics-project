from tensorflow.keras.metrics import AUC
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, MaxPool1D, MaxPool2D, BatchNormalization, Activation, Conv1D
from tensorflow.keras.optimizers import Nadam
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
            CNN: self.create_cnn,
            CNN_2: self.create_cnn_2,
            CNN_3: self.create_cnn_3
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

    def create_cnn_2(self):
        cnn = Sequential([
            Input(shape=(200, 4)),
            Reshape((800, 1)),
            Conv1D(64, kernel_size=5, activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(64, kernel_size=5, activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(64, kernel_size=5, activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            MaxPool1D(pool_size=2),
            Conv1D(64, kernel_size=10, activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            MaxPool1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid")
        ], CNN_2)

        cnn.compile(
            optimizer=Nadam(learning_rate=0.002),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(curve="ROC", name="auroc"),
                AUC(curve="PR", name="auprc")
            ]
        )

        return cnn

    def create_cnn_3(self):
        cnn = Sequential([
            Input(shape=(200, 4)),
            Reshape((200, 4, 1)),
            Conv2D(64, kernel_size=(5, 4), activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
            Conv2D(64, kernel_size=(5, 1), activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
            Conv2D(32, kernel_size=(5, 1), activation="relu"),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ], CNN_3)

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
