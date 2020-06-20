from bioinformatics_project.experiment.experiment_executor import ExperimentExecutor
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow


executor = ExperimentExecutor()
ret = executor.execute_promoters_epigenomic_experiment(DataRetrieval.KEY_PROMOTERS)
executor.print_results(ret)

ret = executor.execute_promoters_epigenomic_experiment(DataRetrieval.KEY_ENHANCERS)
executor.print_results(ret)

