import pandas
from typing import Dict
from matplotlib.pyplot import subplots
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
from termcolor import colored
from tqdm import tqdm

from .data_retrieval import DataRetrieval


class DataChecking:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def check_sample_features_imbalance(self):

        for region, data in self._data.get_epigenomic_data().items():
            rate = data.shape[0] / data.shape[1]
            print(colored(f'The rate between sample and features for {region} is: {rate}',
                          'green' if rate > 10 else 'red'))

    def check_nan_values(self):
        for region, data in self._data.get_epigenomic_data().items():
            print(colored(
                f'For the {region} data, there are {data.isna().values.sum()} NaN values out of {data.values.size} values',
                'green'))

    def check_class_imbalance(self):
        fig, axes = subplots(ncols=2, figsize=(10, 5))

        for axis, (region, y) in zip(axes.ravel(), self._data.get_labels().items()):
            y.hist(color='royalblue', ax=axis, bins=3)
            axis.set_title(f'Number of samples for {region}')
        fig.show()

    def fill_nan_promoters_epigenomic_data(self) -> pandas.DataFrame:
        promoters_epigenomic_data = self._data.get_promoters_epigenomic_data()
        promoters_epigenomic_data = promoters_epigenomic_data.fillna(promoters_epigenomic_data.mean())
        self._data.set_promoters_epigenomic_data(promoters_epigenomic_data)
        return promoters_epigenomic_data

    def check_constant_features(self) -> Dict[str, pandas.DataFrame]:
        ret = {}
        for region, data in self._data.get_epigenomic_data().items():
            ret[region] = data.loc[:, (data != data.iloc[0]).any()]
            if ret[region].shape[1] != data.shape[1]:
                print(colored(f'Features in {region} data are constant, please drop it', 'yellow'))
            else:
                print(colored(f'In {region} data no constant features were found', 'green'))

        return ret

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

    def apply_pearson_correlation(self, p_value_threshold: float = 0.01) -> Dict[str, set]:
        uncorrelated = {}
        for region, data in self._data.get_epigenomic_data().items():
            uncorrelated[region] = set()

            for column in tqdm(data.columns, desc=f'Running Pearson test for {region} data', dynamic_ncols=True, leave=False):
                correlation, p_value = pearsonr(data[column].values.ravel(), self._data.get_labels()[region].values.ravel())
                if p_value > p_value_threshold:
                    uncorrelated[region].add((column, correlation))

            print(f'\rFor {region} data the following uncorrelated feature are found:')
            for column, correlation in uncorrelated[region]:
                print('\r   - ', column, correlation)

        return uncorrelated
