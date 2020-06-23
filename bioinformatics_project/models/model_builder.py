import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.models_type import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input, Flatten
from tensorflow.keras.initializers import Constant, GlorotNormal
from tensorflow.keras.regularizers import l1_l2


class ModelBuilder:

    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_epigenomic_functions(self):
        return {
            DECISION_TREE_GRID: self.create_decision_tree_grid,
            RANDOM_FOREST_GRID: self.create_random_forest_grid,
            PERCEPTRON: self.create_perceptron,
            PERCEPTRON_2: self.create_perceptron,
            MLP: self.create_mlp,
            MLP_2: self.create_mlp_2,
            FFNN: self.create_ffnn,
            FFNN_2: self.create_ffnn_2,
            FFNN_3: self.create_ffnn_3,
            FFNN_4: self.create_ffnn_4,
            FFNN_5: self.create_ffnn_5,
            FFNN_6: self.create_ffnn_6,
            FFNN_7: self.create_ffnn_7,
            FFNN_8: self.create_ffnn_8
        }

    def create_decision_tree_grid(self, _, parameters):
        return DecisionTreeClassifier(**parameters), {}

    def create_random_forest_grid(self, _, parameters):
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

    def create_mlp_2(self, region, parameters):
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

    def create_ffnn(self, region, parameters):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(256, activation="relu"),
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ], "FFNN")

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_2(self, region, parameters):
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
        ], FFNN_2)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_3(self, region, parameters):
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
        ], FFNN_3)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_4(self, region, parameters):
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(64, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Activation("relu"),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(8, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ], FFNN_4)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy"
        )

        return ffnn, parameters

    def create_ffnn_5(self, region, parameters):
        bias = numpy.log([numpy.count_nonzero(
            self._data.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS][1] == True) /
                         numpy.count_nonzero(
                             self._data.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS][1] == False)])
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
            Dense(1, activation="sigmoid", bias_initializer=Constant(bias))
        ], FFNN_5)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
        )

        return ffnn, parameters

    def create_ffnn_6(self, region, parameters):
        bias = numpy.log([numpy.count_nonzero(
            self._data.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS][1] == True) /
                          numpy.count_nonzero(
                              self._data.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS][1] == False)])
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
            Dense(1, activation="sigmoid", bias_initializer=Constant(bias))
        ], FFNN_6)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
        )

        return ffnn, parameters

    def create_ffnn_7(self, region, parameters):
        bias = numpy.log([numpy.count_nonzero(
            self._data.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS][1] == True) /
                          numpy.count_nonzero(
                              self._data.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS][1] == False)])
        ffnn = Sequential([
            Input(shape=(len(self._data.get_epigenomic_data()[region].columns), )),
            Dense(64, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Activation("relu"),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(16, activation="relu"),
            Dropout(0.5),
            Dense(8, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid", bias_initializer=Constant(bias))
        ], FFNN_7)

        ffnn.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
        )

        return ffnn, parameters

    def create_ffnn_8(self, region, parameters):
        Input_layer = Input(shape=(len(self._data.get_epigenomic_data()[region].columns), 1))
        initializer = GlorotNormal()
        x = Dense(units=50, activation='relu', kernel_initializer=initializer)(Input_layer)
        x = Dense(units=200, activation='relu', kernel_initializer=initializer)(x)
        x = Dense(units=50, activation='relu', kernel_initializer=initializer)(x)
        x = Dense(units=200, activation='relu', kernel_initializer=initializer)(x)
        encoded = Dense(units=64, activation='relu', kernel_initializer=initializer)(x)

        autoencoder = Model(Input_layer, encoded)
        autoencoder = Sequential(autoencoder)
        autoencoder.add(Flatten())
        autoencoder.add(Dense(128, activation="relu", kernel_initializer=initializer, activity_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        autoencoder.add(Dense(64, activation="relu", kernel_initializer=initializer, activity_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        autoencoder.add(Dense(32, activation="relu", kernel_initializer=initializer, activity_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        autoencoder.add(Dense(16, activation="relu", kernel_initializer=initializer, activity_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        autoencoder.add(Dense(1, activation="sigmoid"))
        autoencoder.compile(optimizer='Adadelta', loss='binary_crossentropy')

        return autoencoder, parameters
