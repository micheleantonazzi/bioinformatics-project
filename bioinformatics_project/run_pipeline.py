from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
from bioinformatics_project.data_retrieval_and_manipulation.data_checking import DataChecking

data_ret = DataRetrieval()
data_ret.load_promoters_epigenomic_data()
data_ret.load_enhancers_epigenomic_data()
data_ret.load_genome_data()
data_ret.extract_promoters_sequence_data()
data_ret.extract_enhancers_sequence_data()

data_preprocessing = DataPreprocessing(data_ret)
data_preprocessing.drop_constant_features()
data_preprocessing.fill_nan_data()
data_preprocessing.apply_z_scoring()
uncorrelated = data_preprocessing.apply_pearson_spearman_correlation()
uncorrelated, scores = data_preprocessing.apply_mic(uncorrelated)
#data_ret.remove_uncorrelated_features(uncorrelated)
#data_preprocessing.apply_pearson_for_features_correlation()
#data_preprocessing.apply_boruta(5)

data_checking = DataChecking(data_ret)
data_checking.check_sample_features_imbalance()
data_checking.check_class_imbalance()
data_checking.print_correlated_feature(scores)
data_checking.print_features_different_active_inactive()
data_checking.print_pair_features_different()







