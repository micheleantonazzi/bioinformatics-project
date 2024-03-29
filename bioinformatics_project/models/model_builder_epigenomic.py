import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input
from tensorflow.keras.initializers import Constant
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD

"""
    This class is responsible to build the model for the epigenomic experiments
"""
class ModelBuilderEpigenomic:

    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_epigenomic_functions(self):
        return {
            DECISION_TREE_GRID: self.create_decision_tree_grid,
            RANDOM_FOREST_GRID: self.create_random_forest_grid,
            PERCEPTRON: self.create_perceptron,
            MLP: self.create_mlp,
            FFNN_1: self.create_ffnn_1,
            FFNN_2: self.create_ffnn_2,
            FFNN_3: self.create_ffnn_3,
            FFNN_4: self.create_ffnn_4
        }

    def create_decision_tree_grid(self, _, parameters, __=None):
        decision_tree = DecisionTreeClassifier(**parameters)
        setattr(decision_tree, 'name', 'DecisionTree')
        return decision_tree, {}

    def create_random_forest_grid(self, _, parameters, __):
        random_forest = RandomForestClassifier(**parameters, n_jobs=-1)
        setattr(random_forest, 'name', 'RandomForest')
        return random_forest, {}

    def create_perceptron(self, region, parameters, _=None):
        perceptron = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(1, activation="sigmoid")
        ], "Perceptron")

        perceptron.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )
        return perceptron, parameters

    def create_mlp(self, region, parameters, _=None):
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

    def create_mlp_2(self, region, parameters, _=None):
        mlp = Sequential([
             Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
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

    def create_ffnn_1(self, region, parameters, _=None):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation="relu"),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Activation("relu"),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ], FFNN_1)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_2(self, region, parameters, _=None):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Activation("relu"),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(16, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ], FFNN_2)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_3(self, region, parameters, learning_labels):
        bias = numpy.log([numpy.count_nonzero(
            learning_labels == 1) /
                          numpy.count_nonzero(
                              learning_labels == 0)])
        print(bias)

        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Activation("relu"),
            Dense(128, activation='relu'),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(16, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid", bias_initializer=Constant(bias))
        ], FFNN_3)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[AUC(curve='PR', name='pr')]
        )

        return ffnn, parameters

    def create_ffnn_4(self, region, parameters, _=None):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(1, activation="sigmoid")
        ], FFNN_4)

        ffnn.compile(
            optimizer=SGD(learning_rate=0.1, decay=0.01),
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_5(self, region, parameters, _=None):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation='relu', kernel_regularizer=l2(l=0.01)),
            BatchNormalization(),
            Activation("relu"),
            Dense(128, activation='relu', kernel_regularizer=l2(l=0.01)),
            Dropout(0.4),
            Dense(64, activation="relu", kernel_regularizer=l2(l=0.01)),
            Dropout(0.4),
            Dense(32, activation="relu", kernel_regularizer=l2(l=0.01)),
            Dropout(0.4),
            Dense(16, activation="relu", kernel_regularizer=l2(l=0.01)),
            Dropout(0.4),
            Dense(1, activation="sigmoid")
        ], FFNN_5)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_6(self, region, parameters, _=None):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Activation("relu"),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ], FFNN_6)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_7(self, region, parameters, _=None):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            Activation("relu"),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ], FFNN_7)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=[AUC(curve='PR', name='auprc')]
        )

        return ffnn, parameters
