from bioinformatics_project.experiment.experiment_executor import ExperimentExecutor
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing_pipeline import DataPreprocessingPipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

data_retrieval = DataRetrieval()
DataPreprocessingPipeline(data_retrieval).execute_v2()

ret = ExperimentExecutor().execute_sequence_experiment(data_retrieval, DataRetrieval.KEY_PROMOTERS, 2)

