import bioinformatics_project.data_retrieval_manipulation as data_retrieval


def test_download_data():
    promoters_data, enhancers_data = data_retrieval.download_epigenomic_data()
    assert promoters_data['data'].size == 20681163
    assert promoters_data['labels'].size == 99909
    assert enhancers_data['data'].size == 13542561
    assert enhancers_data['labels'].size == 65423
