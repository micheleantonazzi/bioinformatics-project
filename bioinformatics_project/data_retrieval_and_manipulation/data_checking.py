import pandas
from matplotlib.pyplot import subplots
from sklearn.preprocessing import RobustScaler
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
            y.hist(color='royalblue', ax=axis, bins=3)
            axis.set_title(f'Number of samples for {region}')
        fig.show()

    def fill_nan_promoters_epigenomic_data(self) -> pandas.DataFrame:
        promoters_epigenomic_data = self._data.get_promoters_epigenomic_data()
        promoters_epigenomic_data = promoters_epigenomic_data.fillna(promoters_epigenomic_data.mean())
        self._data.set_promoters_epigenomic_data(promoters_epigenomic_data)
        return promoters_epigenomic_data

    def check_constant_features(self):
        promoters_epigenomic_data = self._data.get_promoters_epigenomic_data()
        enhancers_epigenomic_data = self._data.get_enhancers_epigenomic_data()

        promoters_data_dropped = promoters_epigenomic_data.loc[:, (promoters_epigenomic_data != promoters_epigenomic_data.iloc[0]).any()]
        if promoters_epigenomic_data.shape[1] != promoters_data_dropped.shape[1]:
            self._data.set_promoters_epigenomic_data(promoters_data_dropped)
            print(colored('Features in promoters data are constant and had to be dropped', 'yellow'))
        else:
            print(colored('In promoters data no constant features were found'))

        enhancers_data_dropped = enhancers_epigenomic_data.loc[:, (enhancers_epigenomic_data != enhancers_epigenomic_data.iloc[0]).any()]
        if enhancers_epigenomic_data.shape[1] != enhancers_data_dropped.shape[1]:
            self._data.set_enhancers_epigenomic_data(enhancers_data_dropped)
            print(colored('Features in enhancers data are constant and had to be dropped', 'yellow', 'green'))
        else:
            print(colored('In enhancers data no constant features were found', 'green'))

    def apply_z_scoring(self):
        promoters_data = self._data.get_promoters_epigenomic_data()
        self._data.set_promoters_epigenomic_data(
            pandas.DataFrame(
                RobustScaler().fit_transform(promoters_data.values),
                columns=promoters_data.columns,
                index=promoters_data.index
            )
        )

        enhancers_data = self._data.get_enhancers_epigenomic_data()
        self._data.set_enhancers_epigenomic_data(
            pandas.DataFrame(
                RobustScaler().fit_transform(enhancers_data.values),
                columns=enhancers_data.columns,
                index=enhancers_data.index
            )
        )
