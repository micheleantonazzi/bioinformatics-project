import bioinformatics_project.data_retrieval_manipulation as data_retrieval


def test_download_data():
    promoters_data, promoters_labels, enhancers_data, enhancers_labels = data_retrieval.download_epigenomic_data()
    assert promoters_data.size == 20681163
    assert promoters_labels.size == 99909
    assert enhancers_data.size == 13542561
    assert enhancers_labels.size == 65423
