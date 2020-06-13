from pytest import fail
from bioinformatics_project.data_retrieval_and_manipulation.data_checking import DataChecking
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval

data_retrieval = DataRetrieval()


def test_load_promoters_epigenomic_data():
    promoters_data = data_retrieval.load_promoters_epigenomic_data()
    assert promoters_data == data_retrieval.get_promoters_data()
    assert len(data_retrieval.get_promoters_epigenomic_data()) == 99909
    assert len(data_retrieval.get_promoters_labels()) == 99909


def test_load_enhancers_epigenomic_data():
    enhancers_data = data_retrieval.load_enhancers_epigenomic_data()
    assert enhancers_data == data_retrieval.get_enhancers_data()
    assert len(data_retrieval.get_enhancers_labels()) == 65423
    assert len(data_retrieval.get_enhancers_epigenomic_data()) == 65423


def test_load_genome_data():
    genome = data_retrieval.load_genome_data()
    assert len(genome) == 25


def test_extract_promoters_sequence_data():
    promoters_sequence_data = data_retrieval.extract_promoters_sequence_data(10)
    assert len(promoters_sequence_data) == 10


def test_extract_enhancers_sequence_data():
    enhancers_sequence_data = data_retrieval.extract_enhancers_sequence_data(5)
    assert len(enhancers_sequence_data) == 5


def test_check_sample_features_imbalance():
    try:
        DataChecking(data_retrieval).check_sample_features_imbalance()
    except:
        fail('Unexpected exception')


def test_check_nan_values():
    try:
        DataChecking(data_retrieval).check_nan_values()
    except:
        fail('Unexpected exception')


def test_fill_nan():
    promoters_data = data_retrieval.get_promoters_epigenomic_data()
    data_checking = DataChecking(data_retrieval)
    assert promoters_data.isna().values.sum() == 1
    assert data_checking.fill_nan_promoters_epigenomic_data().isna().values.sum() == data_retrieval.get_promoters_epigenomic_data().isna().values.sum() == 0


def test_check_class_imbalance():
    try:
        DataChecking(data_retrieval).check_class_imbalance()
    except:
        fail('Unexpected exception')


def test_constant_features():
    try:
        DataChecking(data_retrieval).check_constant_features()
    except:
        fail('Unexpected exception')


def test_z_scoring():
    try:
        DataChecking(data_retrieval).apply_z_scoring()
    except:
        fail('Unexpected exception')


def test_correlation():
    try:
        data_checking = DataChecking(data_retrieval)

        pearson = data_checking.apply_pearson_correlation()
        assert len(pearson[DataRetrieval.KEY_PROMOTERS]) + len(pearson[DataRetrieval.KEY_ENHANCERS]) == 37

        spearman = data_checking.apply_spearman_correlation()
        assert len(spearman[DataRetrieval.KEY_PROMOTERS]) + len(spearman[DataRetrieval.KEY_ENHANCERS]) == 33

        mic = data_checking.apply_mic(data_checking.apply_pearson_spearman_correlation())
        assert len(mic['promoters']) + len(mic['enhancers']) == 44

    except:
        fail('Unexpected exception')
