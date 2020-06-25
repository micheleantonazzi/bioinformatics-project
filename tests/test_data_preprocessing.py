from pytest import fail
from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
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


def test_constant_features():
    try:
        DataPreprocessing(data_retrieval).drop_constant_features()
    except:
        fail('Unexpected exception')


def test_fill_nan():
    promoters_data = data_retrieval.get_promoters_epigenomic_data()
    data_pre = DataPreprocessing(data_retrieval)
    assert promoters_data.isna().values.sum() == 1
    data_pre.fill_nan_data()
    assert data_retrieval.get_promoters_epigenomic_data().isna().values.sum() == 0


def test_z_scoring():
    try:
        DataPreprocessing(data_retrieval).apply_z_scoring()
    except:
        fail('Unexpected exception')


def test_pearson_spearman_correlation():
    try:
        data_pre = DataPreprocessing(data_retrieval)

        pearson = data_pre.apply_pearson_correlation()
        assert len(pearson[DataRetrieval.KEY_PROMOTERS]) + len(pearson[DataRetrieval.KEY_ENHANCERS]) == 37

        spearman = data_pre.apply_spearman_correlation()
        assert len(spearman[DataRetrieval.KEY_PROMOTERS]) + len(spearman[DataRetrieval.KEY_ENHANCERS]) == 33
    except:
        fail('Unexpected exception')


def test_mic_and_remove_features():
    try:
        data_pre = DataPreprocessing(data_retrieval)
        uncorrelated = data_pre.apply_pearson_spearman_correlation()
        assert len(uncorrelated['promoters']) + len(uncorrelated['enhancers']) == 44
        uncorrelated = {region: list(data)[:1] for region, data in uncorrelated.items()}
        mic = data_pre.apply_mic_on_selected_features(uncorrelated)
        assert len(mic['promoters']) + len(mic['enhancers']) == 2

        promoters_data_columns = len(data_retrieval.get_promoters_epigenomic_data().columns)
        enhancers_data_columns = len(data_retrieval.get_enhancers_epigenomic_data().columns)

        data_retrieval.remove_uncorrelated_features(mic)
        assert promoters_data_columns == len(data_retrieval.get_promoters_epigenomic_data().columns) + len(
            mic['promoters'])
        assert enhancers_data_columns == len(data_retrieval.get_enhancers_epigenomic_data().columns) + len(
            mic['enhancers'])
    except Exception as e:
        fail('Unexpected exception', e)


def test_feature_feature_correlation():
    try:
        extremely_correlated, scores = DataPreprocessing(data_retrieval).apply_pearson_for_features_correlation()
        assert len(extremely_correlated[DataRetrieval.KEY_PROMOTERS]) == \
               len(extremely_correlated[DataRetrieval.KEY_ENHANCERS]) == 0
    except:
        fail('Unexpected exception')
