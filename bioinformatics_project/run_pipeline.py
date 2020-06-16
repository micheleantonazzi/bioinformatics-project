from bioinformatics_project.data_retrieval_and_manipulation.data_checking import DataChecking
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval

data_ret = DataRetrieval()
data_ret.load_promoters_epigenomic_data()
data_ret.load_enhancers_epigenomic_data()
data_ret.load_genome_data()
data_ret.extract_promoters_sequence_data()
data_ret.extract_enhancers_sequence_data()

data_checking = DataChecking(data_ret)
data_checking.check_sample_features_imbalance()
data_checking.check_nan_values()
data_checking.fill_nan_promoters_epigenomic_data()
data_checking.check_constant_features()
data_checking.apply_tsne()

data_checking.apply_z_scoring()






