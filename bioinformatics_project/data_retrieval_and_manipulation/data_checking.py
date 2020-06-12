from matplotlib.pyplot import subplots
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

    def check_nan_values(self):
        promoters_epigenomic_data = self._data.get_promoters_epigenomic_data()

        print(colored(
            f'For the promoters data, there are {promoters_epigenomic_data.isna().values.sum()} NaN values out of {promoters_epigenomic_data.values.size} values',
            'green'))

        enhancers_epigenomic_data = self._data.get_enhancers_epigenomic_data()
        print(colored(
            f'In the document there are {enhancers_epigenomic_data.isna().values.sum()} NaN values out of {enhancers_epigenomic_data.values.size} values.',
            'green'))

    def check_class_imbalance(self):
        fig, axes = subplots(ncols=2, figsize=(10, 5))

        for axis, (region, y) in zip(axes.ravel(), [('promoters', self._data.get_promoters_labels()),
                                                    ('enhancers', self._data.get_enhancers_labels())]):
            y.hist(ax=axis, bins=3)
            axis.set_title(f"Classes count in {region}")
        fig.show()
