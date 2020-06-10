import bioinformatics_project.data_retrieval_manipulation as data_retrieval


def test_download_data():
    promoters_data, enhancers_data = data_retrieval.download_epigenomic_data()
    assert promoters_data['data'].size == 20681163
    assert promoters_data['labels'].size == 99909
    assert enhancers_data['data'].size == 13542561
    assert enhancers_data['labels'].size == 65423


def test_download_sequence_data():
    hg19 = data_retrieval.download_genome_data()
    assert hg19.__sizeof__() == 32


def test_extract_sequence_data():
    promoters_data, enhancers_data = data_retrieval.download_epigenomic_data()
    hg19 = data_retrieval.download_genome_data()
    promoters_sequence_data = data_retrieval.extract_sequence_data(hg19, promoters_data["data"])
    assert promoters_sequence_data.size == 499545


def test_flat_one_not_encoding():
    promoters_data, enhancers_data = data_retrieval.download_epigenomic_data()
    hg19 = data_retrieval.download_genome_data()
    promoters_sequence_one_not = data_retrieval.sequence_data_flat_one_not_encoded(hg19, promoters_data['data'], 200,
                                                                                   'actg')
    assert len(promoters_sequence_one_not) == 99909
