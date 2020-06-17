from typing import Dict

import numpy
import pandas
import seaborn
from matplotlib.pyplot import subplots, show
from sklearn.metrics import euclidean_distances
from termcolor import colored
from prince import MFA
from tqdm import tqdm
from tsnecuda import TSNE
from sklearn.decomposition import PCA


from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class DataChecking:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def check_sample_features_imbalance(self):
        for region, data in self._data.get_epigenomic_data().items():
            rate = data.shape[0] / data.shape[1]
            print(colored(f'The rate between sample and features for {region} is: {rate}',
                          'green' if rate > 10 else 'red'))

    def check_class_imbalance(self):
        fig, axes = subplots(ncols=2, figsize=(10, 5))

        for axis, (region, y) in zip(axes.ravel(), self._data.get_labels().items()):
            y.hist(color='royalblue', ax=axis, bins=3)
            axis.set_title(f'Number of samples for {region}')
        fig.show()

    def print_correlated_feature(self, features: Dict[str, list]):
        features = {
            region: sorted(score, key=lambda x: numpy.abs(x[0]), reverse=True)
            for region, score in features.items()
        }

        for region, data in self._data.get_epigenomic_data().items():
            _, firsts, seconds = list(zip(*features[region][:2]))
            columns = list(set(firsts + seconds))
            grid = seaborn.pairplot(pandas.concat([
                data[columns],
                self._data.get_labels()[region],
            ], axis=1), hue=self._data.get_labels()[region].columns[0])
            grid.fig.suptitle(f'Most correlated features for {region}')
            show()

            _, firsts, seconds = list(zip(*features[region][-2:]))
            columns = list(set(firsts + seconds))
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

                cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
                cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

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

    def cannylab_tsne(self, data: numpy.ndarray, perplexity: int, dimensionality_threshold: int = 50):
        if data.shape[1] > dimensionality_threshold:
            data = self.pca(data, components=dimensionality_threshold)
        return TSNE(perplexity=perplexity, random_seed=42).fit_transform(data)

    def _get_data_decomposition_task(self):
        return {
            "x": [
                *[
                    val
                    for val in self._data.get_epigenomic_data().values()
                ],
                *[
                    val
                    for val in self._data.get_sequence_data().values()
                ],
                pandas.concat(self._data.get_sequence_data().values()),
                pandas.concat(self._data.get_sequence_data().values()),
            ],
            "y": [
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
            "titles": [
                'Epigenomes promoters',
                'Epigenomes enhancers',
                'Sequences promoters',
                'Sequences enhancers',
                'Sequences active regions',
                'Sequences regions types',
            ]
        }

    def apply_pca(self, ):
        tasks = self._get_data_decomposition_task()
        xs = tasks["x"]
        ys = tasks["y"]
        titles = tasks["titles"]
        colors = numpy.array([
            "tab:blue",
            "tab:orange",
        ])
        fig, axes = subplots(nrows=2, ncols=3, figsize=(32, 16))

        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
            axis.scatter(*self.pca(x).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"PCA decomposition - {title}", fontdict={'fontsize': 25,
                                                                     'fontweight': 25,
                                                                     'verticalalignment': 'baseline',
                                                                     'horizontalalignment': 'center'})
        show()

    def apply_mfa(self):
        tasks = {
            "x": [
                *[
                    val
                    for val in self._data.get_sequence_data().values()
                ],

            ],
            "y": [
                *[
                    val.values.ravel()
                    for val in self._data.get_labels().values()
                ],
            ],
            "titles": [
                'Sequences promoters',
                'Sequences enhancers',
            ]
        }
        xs = tasks["x"]
        ys = tasks["y"]
        titles = tasks["titles"]
        colors = numpy.array([
            "tab:blue",
            "tab:orange",
        ])
        fig, axes = subplots(nrows=1, ncols=2, figsize=(32, 16))

        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing MFAs", total=len(xs)):
            axis.scatter(*self.mfa(x).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"MFA decomposition - {title}")
        show()

    def apply_cannylab_tsne(self):
        tasks = self._get_data_decomposition_task()
        xs = tasks["x"]
        ys = tasks["y"]
        titles = tasks["titles"]
        colors = numpy.array([
            "tab:blue",
            "tab:orange",
        ])
        for perpexity in tqdm((30, 40, 50, 100, 500, 5000), desc="Running perplexities"):
            fig, axes = subplots(nrows=2, ncols=3, figsize=(40, 20))
            for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing TSNEs", total=len(xs)):
                axis.scatter(*self.cannylab_tsne(x, perplexity=perpexity).T, s=1, color=colors[y])
                axis.xaxis.set_visible(False)
                axis.yaxis.set_visible(False)
                axis.set_title(f"TSNE decomposition - {title}", fontdict={'fontsize': 25,
                                                                          'fontweight': 25,
                                                                          'verticalalignment': 'baseline',
                                                                          'horizontalalignment': 'center'})
            fig.tight_layout()
            show()

