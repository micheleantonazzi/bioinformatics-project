import numpy
import pandas
import seaborn
from typing import Dict
from matplotlib.pyplot import subplots, show, text
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import RobustScaler
from termcolor import colored
from tqdm import tqdm
from minepy import MINE

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

            for column in tqdm(data.columns, desc=f'Running Pearson test for {region} data', dynamic_ncols=True,
                               leave=False):
                correlation, p_value = pearsonr(data[column].values.ravel(),
                                                self._data.get_labels()[region].values.ravel())
                if p_value > p_value_threshold:
                    uncorrelated[region].add(column)

            print(colored(f'\rApplied Pearson test for {region}, {len(uncorrelated[region])} features are found',
                          'green'))

        return uncorrelated

    def apply_spearman_correlation(self, p_value_threshold: float = 0.01) -> Dict[str, set]:
        uncorrelated = {}
        for region, data in self._data.get_epigenomic_data().items():
            uncorrelated[region] = set()

            for column in tqdm(data.columns, desc=f'Running Spearman test for {region} data', dynamic_ncols=True,
                               leave=False):
                correlation, p_value = spearmanr(data[column].values.ravel(),
                                                 self._data.get_labels()[region].values.ravel())
                if p_value > p_value_threshold:
                    uncorrelated[region].add(column)

            print(colored(f'\rApplied Spearman test for {region}, {len(uncorrelated[region])} features are found',
                          'green'))

        return uncorrelated

    def apply_pearson_spearman_correlation(self, pearson_threshold: float = 0.01,
                                           spearman_threshold: float = 0.01) -> Dict[str, set]:
        uncorrelated = {}
        for region, data in self._data.get_epigenomic_data().items():
            uncorrelated[region] = set()

            for column in tqdm(data.columns, desc=f'Running Spearman test for {region} data', dynamic_ncols=True,
                               leave=False):
                correlation, p_value = spearmanr(data[column].values.ravel(),
                                                 self._data.get_labels()[region].values.ravel())
                if p_value > spearman_threshold:
                    uncorrelated[region].add(column)

                correlation, p_value = pearsonr(data[column].values.ravel(),
                                                self._data.get_labels()[region].values.ravel())
                if p_value > pearson_threshold:
                    uncorrelated[region].add(column)

            print(colored(
                f'\rApplied Pearson and Spearman test for {region}, {len(uncorrelated[region])} features are found',
                'green'))

        return uncorrelated

    def apply_mic(self, uncorrelated: Dict[str, set], correlation_threshold: float = 0.05) -> Dict[str, set]:
        for region, data in self._data.get_epigenomic_data().items():
            for column in tqdm(uncorrelated[region], desc=f'Running MIC test for {region} data', dynamic_ncols=True,
                               leave=False):
                mine = MINE()
                mine.compute_score(data[column].values.ravel(), self._data.get_labels()[region].values.ravel())
                score = mine.mic()

                if score >= correlation_threshold:
                    uncorrelated[region].remove(column)

            print(colored(f'\rApplied MIC test for {region}, {len(uncorrelated[region])} features are found', 'green'))

        return uncorrelated

    def apply_pearson_for_features_correlation(self, pearson_threshold: float = 0.01,
                                               correlation_threshold: float = 0.95) -> (Dict[str, set],
                                                                                        Dict[str, list]):
        extremely_correlated = {
            region: set()
            for region in [DataRetrieval.KEY_PROMOTERS, DataRetrieval.KEY_ENHANCERS]
        }

        scores = {
            region: []
            for region in self._data.get_epigenomic_data()
        }

        for region, data in self._data.get_epigenomic_data().items():
            for i, column in tqdm(
                    enumerate(data.columns),
                    total=len(data.columns), desc=f"Running Pearson test for {region}", dynamic_ncols=True,
                    leave=False):
                for feature in data.columns[i + 1:]:
                    correlation, p_value = pearsonr(data[column].values.ravel(), data[feature].values.ravel())
                    correlation = numpy.abs(correlation)
                    scores[region].append((correlation, column, feature))
                    if p_value < pearson_threshold and correlation > correlation_threshold:
                        print(region, column, feature, correlation)
                        if entropy(data[column]) > entropy(data[feature]):
                            extremely_correlated[region].add(feature)
                        else:
                            extremely_correlated[region].add(column)

            print(
                colored(f'\rApplied Pearson for features correlation for {region}, {len(extremely_correlated[region])}'
                        f' useless features are found', 'green'))

        return extremely_correlated, scores

    def print_scatter_plot(self, features: Dict[str, list]):
        features = {
            region: sorted(score, key=lambda x: numpy.abs(x[0]), reverse=True)
            for region, score in features.items()
        }

        for region, data in self._data.get_epigenomic_data().items():
            _, firsts, seconds = list(zip(*features[region][:2]))
            columns = list(set(firsts+seconds))
            grid = seaborn.pairplot(pandas.concat([
                data[columns],
                self._data.get_labels()[region],
            ], axis=1), hue=self._data.get_labels()[region].columns[0])
            grid.fig.suptitle(f'Most correlated features for {region}')
            show()

            _, firsts, seconds = list(zip(*features[region][-2:]))
            columns = list(set(firsts+seconds))
            grid = seaborn.pairplot(pandas.concat([
                data[columns],
                self._data.get_labels()[region],
            ], axis=1), hue=self._data.get_labels()[region].columns[0])
            grid.fig.suptitle(f'Most uncorrelated features for {region}')
            show()

    def print_feature_distributions(self, features_number: int = 5):
        for region, data in self._data.get_epigenomic_data().items():
            distance = euclidean_distances(data.T)
            most_distance_columns_indices = numpy.argsort(-numpy.mean(distance, axis=1).flatten())[:features_number]
            columns = data.columns[most_distance_columns_indices]
            fig, axes = subplots(nrows=1, ncols=features_number, figsize=(25, 5))
            for column, axis in zip(columns, axes.flatten()):
                head, tail = data[column].quantile([0.05, 0.95]).values.ravel()

                mask = ((data[column] < tail) & (data[column] > head)).values

                cleared_x = data[column][mask]
                cleared_y = self._data.get_labels()[region].values.ravel()[mask]

                cleared_x[cleared_y==0].hist(ax=axis, bins=20)
                cleared_x[cleared_y==1].hist(ax=axis, bins=20)

                axis.set_title(column)
            show()

