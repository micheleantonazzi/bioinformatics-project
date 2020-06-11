from termcolor import colored

from .data_retrieval import DataRetrieval


class DataChecking:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def check_sample_features_imbalance(self):
        promoters_epigenomic_data = self._data.get_promoters_epigenomic_data()
        enhancers_epigenomic_data = self._data.get_enhancers_epigenomic_data()

        promoters_rate = promoters_epigenomic_data.shape[0] / promoters_epigenomic_data.shape[1]
        print(colored('The rate between sample and features for promoters is: ' + str(int(promoters_rate)),
                      'green' if promoters_rate > 10 else 'red'))

        enhancers_rate = enhancers_epigenomic_data.shape[0] / enhancers_epigenomic_data.shape[1]
        print(colored('The rate between sample and features for enhancers is: ' + str(int(enhancers_rate)),
                      'green' if enhancers_rate > 10 else 'red'))
