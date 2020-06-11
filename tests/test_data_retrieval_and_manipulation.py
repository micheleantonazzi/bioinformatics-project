import pytest
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval

data_retrieval = DataRetrieval()


@pytest.mark.dependency()
def test_load_promoters_epigenomic_data():
    promoters_data = data_retrieval.load_promoters_epogenomic_data()
    assert len(promoters_data[data_retrieval.key_epigenomic]) == 99909
    assert len(promoters_data[data_retrieval.key_labels]) == 99909


@pytest.mark.dependency()
def test_load_enhancers_epigenomic_data():
    enhancers_data = data_retrieval.load_enhancers_epigenomic_data()
    assert len(enhancers_data[data_retrieval.key_epigenomic]) == 65423
    assert len(enhancers_data[data_retrieval.key_labels]) == 65423


@pytest.mark.dependency()
def test_load_genome_data():
    genome = data_retrieval.load_genome_data()
    assert len(genome) == 25


@pytest.mark.dependency(depends=['test_load_promoters_epigenomic_data', 'test_load_genome_data'])
def test_extract_promoters_sequence_data():
    promoters_sequence_data = data_retrieval.extract_promoters_sequence_data()
    assert len(promoters_sequence_data) == 99909


@pytest.mark.dependency(depends=['test_load_enhancers_epigenomic_data', 'test_load_genome_data'])
def test_extract_enhancers_sequence_data():
    enhancers_sequence_data = data_retrieval.extract_enhancers_sequence_data()
    assert len(enhancers_sequence_data) == 65423
