from tensorflow.keras.callbacks import EarlyStopping
from termcolor import colored

from bioinformatics_project.models.models_type import *
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class ParameterSelectorSequence:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def get_sequence_functions(self):
        return {
            PERCEPTRON_SEQUENCE: self.get_perceptron_parameters
        }

    def get_perceptron_parameters(self):
        parameters = dict(
            epochs=1000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ]
        )

        best_parameters = {DataRetrieval.KEY_PROMOTERS: parameters, DataRetrieval.KEY_ENHANCERS: parameters}
        for region, data in best_parameters.items():
            print(colored(f'Best {PERCEPTRON_SEQUENCE} parameters for {region}: ' + str(data), 'green'))
        return best_parameters
