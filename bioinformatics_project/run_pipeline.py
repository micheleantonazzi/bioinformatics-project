from bioinformatics_project.experiment.experiment_executor import ExperimentExecutor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow


ret = ExperimentExecutor().execute_promoters_epigenomic_experiment()

