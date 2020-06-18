from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
from bioinformatics_project.data_retrieval_and_manipulation.data_checking import DataChecking
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing_pipeline import DataPreprocessingPipeline

data_retrieval = DataRetrieval()
data_retrieval.load_enhancers_epigenomic_data()
data_retrieval.load_promoters_epigenomic_data()

data_preprocessing = DataPreprocessing(data_retrieval)
data_preprocessing.fill_nan_data()
data_preprocessing.apply_z_scoring()
_, scores = data_preprocessing.apply_spearman_for_features_correlation()
DataChecking(data_retrieval).print_correlated_feature(scores, 'Spearman')






