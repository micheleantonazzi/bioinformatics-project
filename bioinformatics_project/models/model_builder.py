from keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.callbacks import EarlyStopping

from bioinformatics_project.models.models_type import *


class ModelBuilder:

    def __init__(self):
        pass

    def create_module(self, model_type: str):
        if model_type == DECISION_TREE:
            return DecisionTreeClassifier()

        if model_type == PERCEPTRON:
            perceptron = Sequential([
                Input(shape=(191, )),
                Dense(1, activation="sigmoid")
            ], 'Perceptron')

            perceptron.compile(
                optimizer='nadam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            parameters = dict(
                epochs=1000,
                batch_size=1024,
                validation_split=0.1,
                shuffle=True,
                verbose=False,
                callbacks=[
                    EarlyStopping(monitor='val_loss', mode='min', patience=50),
                    TQDMNotebookCallback(leave_outer=False)
                ]
            )

            return KerasClassifier(
                build_fn=lambda: perceptron,
                epochs=1000,
                batch_size=1024,
                validation_split=0.1,
                shuffle=True,
                verbose=False,
                callbacks=[
                    EarlyStopping(monitor='val_loss', mode='min', patience=50),
                    TQDMNotebookCallback(leave_outer=False)
                ]
            )


    def create_model(self, type: str, parameters):
        if type == DECISION_TREE:
            return DecisionTreeClassifier(
                **parameters
            )
