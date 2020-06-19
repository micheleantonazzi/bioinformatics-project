from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
from bioinformatics_project.data_retrieval_and_manipulation.data_checking import DataChecking
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing_pipeline import DataPreprocessingPipeline
from bioinformatics_project.models.parameter_selector import ParameterSelector
from bioinformatics_project.models.models_type import *

data_retrieval = DataRetrieval()
DataPreprocessingPipeline(data_retrieval).execute_v2()

functions = ParameterSelector(data_retrieval).get_functions()
for model, function in functions.items():
    function()

