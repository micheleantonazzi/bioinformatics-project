from multiprocessing import cpu_count

import numpy
import pandas
import seaborn
from typing import Dict

from boruta import BorutaPy
from matplotlib.pyplot import subplots, show
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import RobustScaler
from termcolor import colored
from tqdm import tqdm
from minepy import MINE
from prince import MFA

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

    def print_features_different_active_inactive(self, features_number: int = 5):
        for region, data in self._data.get_epigenomic_data().items():
            distance = euclidean_distances(data.T)
            most_distance_columns_indices = numpy.argsort(-numpy.mean(distance, axis=1).flatten())[:features_number]
            columns = data.columns[most_distance_columns_indices]
            fig, axes = subplots(nrows=1, ncols=features_number, figsize=(5 * features_number, 5))
            for column, axis in zip(columns, axes.flatten()):
                head, tail = data[column].quantile([0.05, 0.95]).values.ravel()

                mask = ((data[column] < tail) & (data[column] > head)).values

                cleared_x = data[column][mask]
                cleared_y = self._data.get_labels()[region].values.ravel()[mask]

                cleared_x[cleared_y==0].hist(ax=axis, bins=20)
                cleared_x[cleared_y==1].hist(ax=axis, bins=20)

                axis.set_title(column)
            show()

    def print_pair_features_different(self, tuples_number: int = 5):
        for region, data in self._data.get_epigenomic_data().items():
            distance = euclidean_distances(data.T)
            dist = numpy.triu(distance)
            tuples = list(zip(*numpy.unravel_index(numpy.argsort(-dist.ravel()), dist.shape)))[:tuples_number]
            fig, axes = subplots(nrows=1, ncols=tuples_number, figsize=(5 * tuples_number, 5))
            for (i, j), axis in zip(tuples, axes.flatten()):
                column_i = data.columns[i]
                column_j = data.columns[j]
                for column in (column_i, column_j):
                    head, tail = data[column].quantile([0.05, 0.95]).values.ravel()
                    mask = ((data[column] < tail) & (data[column] > head)).values
                    data[column][mask].hist(ax=axis, bins=20, alpha=0.5)
                axis.set_title(f"{column_i} and {column_j}")
            fig.tight_layout()
            show()

    def apply_boruta(self, max_iter: int = 10, threshold: float = 0.05, max_depth: int = 5):
        irrelevant_features = {}
        for region, data in tqdm(
                self._data.get_epigenomic_data().items(),
                desc="Running Baruta Feature estimation"
        ):
            boruta_selector = BorutaPy(
                RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=max_depth),
                n_estimators='auto',
                verbose=2,
                alpha=threshold,
                max_iter=max_iter,
                random_state=42
            )
            irrelevant_features[region] = boruta_selector.fit(data.values, self._data.get_labels()[region].values.ravel()).transform(data.values)
        return irrelevant_features

    def pca(self, data: pandas.DataFrame, components: int = 2) -> numpy.ndarray:
        return PCA(n_components=components, random_state=42).fit_transform(data)

    def mfa(self, data: pandas.DataFrame, components: int = 2, nucleotides: str = 'actg') -> numpy.ndarray:
        return MFA(groups={
            nucleotide: [
                column
                for column in data.columns
                if nucleotide in column
            ]
            for nucleotide in nucleotides
        }, n_components=components, random_state=42).fit_transform(data)

    def _get_data_decomposition_task(self):
        return {
            "x":[
                       *[
                           val.values
                           for val in self._data.get_epigenomic_data().values()
                       ],
                       *[
                           val.values
                           for val in self._data.get_sequence_data().values()
                       ],
                       pandas.concat(self._data.get_sequence_data().values()).values,
                       pandas.concat(self._data.get_sequence_data().values()).values,
                   ],
            "y":[
                *[
                    val.values.ravel()
                    for val in self._data.get_labels().values()
                ],
                *[
                    val.values.ravel()
                    for val in self._data.get_labels().values()
                ],
                pandas.concat(self._data.get_labels().values()).values.ravel(),
                numpy.vstack([numpy.ones_like(self._data.get_labels()[DataRetrieval.KEY_PROMOTERS]),
                               numpy.zeros_like(self._data.get_labels()[DataRetrieval.KEY_ENHANCERS])]).ravel(),
            ],
            "titles":[
                'Epigenomes promoters',
                'Epigenomes enhancers',
                'Sequences promoters',
                'Sequences enhancers',
                'Sequences active regions',
                'Sequences regions types',
            ]
        }

    def apply_pca(self,):
        tasks = self._get_data_decomposition_task()
        xs = tasks["x"]
        ys = tasks["y"]
        titles = tasks["titles"]
        colors = numpy.array([
            "tab:blue",
            "tab:orange",
        ])
        fig, axes = subplots(nrows=2, ncols=4, figsize=(32, 16))

        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
            axis.scatter(*self.pca(x).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"PCA decomposition - {title}")
        show()
