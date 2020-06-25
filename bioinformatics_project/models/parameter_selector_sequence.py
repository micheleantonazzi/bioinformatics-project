from tensorflow.keras.callbacks import EarlyStopping
from termcolor import colored

from bioinformatics_project.models.models_type import *
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class ParameterSelectorSequence:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_sequence_functions(self):
        return {
            PERCEPTRON_SEQUENCE: self.get_perceptron_parameters,
            MLP_SEQUENCE: self.get_mlp_parameters,
            FFNN_SEQUENCE: self.get_ffnn_parameters,
            CNN: self.get_cnn_parameters,
            CNN_2: self.get_cnn_2_parameters,
            CNN_3: self.get_cnn_3_parameters
        }

    def get_perceptron_parameters(self):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {PERCEPTRON_SEQUENCE} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_mlp_parameters(self):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {MLP_SEQUENCE} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_ffnn_parameters(self):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {FFNN_SEQUENCE} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_cnn_parameters(self):
        parameters = dict(
            epochs=100,
            batch_size=1024,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=10),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {CNN} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_cnn_2_parameters(self):
        parameters = dict(
            epochs=100,
            batch_size=1024,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=10),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {CNN_2} parameters for {region}: ' + str(data), 'green'))
        return best_parameters

    def get_cnn_3_parameters(self):
        parameters = dict(
            epochs=100,
            batch_size=1024,
            shuffle=True,
            verbose=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=10),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {CNN_3} parameters for {region}: ' + str(data), 'green'))
        return best_parameters
