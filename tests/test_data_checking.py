from pytest import fail

from bioinformatics_project.data_retrieval_and_manipulation.data_checking import DataChecking
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing

data_retrieval = DataRetrieval()
data_retrieval.load_promoters_epigenomic_data()
data_retrieval.load_enhancers_epigenomic_data()
data_retrieval.load_genome_data()
data_retrieval.extract_promoters_sequence_data(5)
data_retrieval.extract_enhancers_sequence_data(10)

data_preprocessing = DataPreprocessing(data_retrieval)
data_preprocessing.drop_constant_features()
data_preprocessing.fill_nan_data()
data_preprocessing.apply_z_scoring()
correlated, scores = data_preprocessing.apply_pearson_for_features_correlation()


def test_check_sample_features_imbalance():
    try:
        DataChecking(data_retrieval).check_sample_features_imbalance()
    except:
        fail('Unexpected exception')


def test_check_class_imbalance():
    try:
        DataChecking(data_retrieval).check_class_imbalance()
    except:
        fail('Unexpected exception')


def test_print_correlated_feature():
    try:
        DataChecking(data_retrieval).print_correlated_feature(scores)
    except:
        fail('Unexpected exception')


def test_print_features_different_active_inactive():
    try:
        DataChecking(data_retrieval).print_features_different_active_inactive()
    except:
        fail('Unexpected exception')


def test_print_pair_features_different():
    try:
        DataChecking(data_retrieval).print_pair_features_different()
    except:
        fail('Unexpected exception')


def test_pca():
    try:
        DataChecking(data_retrieval).apply_pca()
    except:
        fail('Unexpected exception')

