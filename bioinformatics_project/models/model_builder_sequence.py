from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

from bioinformatics_project.models.models_type import *
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class ModelBuilderSequence:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_sequence_functions(self):
        return {
            PERCEPTRON_SEQUENCE: self.create_perceptron
        }

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
