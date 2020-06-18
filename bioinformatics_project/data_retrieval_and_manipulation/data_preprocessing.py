from multiprocessing import cpu_count

import numpy
import pandas
from typing import Dict
from boruta import BorutaPy
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from termcolor import colored
from tqdm import tqdm
from minepy import MINE

from .data_retrieval import DataRetrieval


class DataPreprocessing:
    def __init__(self, data: DataRetrieval):
        self._data = data

    def fill_nan_data(self):
        for region, data in self._data.get_epigenomic_data().items():
            nan = data.isna().values.sum()
            if nan > 0:
                print(colored(f'In {region} data are found {nan} NaN values, they are substituted with the mean',
                              'yellow'))
                new_data = data.fillna(data.mean())
                self._data.set_epigenomic_data(region, new_data)
            else:
                print(colored(f'In {region} data are found {nan} NaN values', 'green'))

    def drop_constant_features(self):
        for region, data in self._data.get_epigenomic_data().items():
            new_data = data.loc[:, (data != data.iloc[0]).any()]
            if new_data.shape[1] != data.shape[1]:
                print(colored(f'Features in {region} data are constant, they are dropped', 'yellow'))
                self._data.set_epigenomic_data(region, new_data)
            else:
                print(colored(f'In {region} data no constant features were found', 'green'))

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

    def apply_mic_on_selected_features(self, uncorrelated: Dict[str, set], correlation_threshold: float = 0.05) -> Dict[
        str, set]:
        for region, data in self._data.get_epigenomic_data().items():
            for column in tqdm(uncorrelated[region], desc=f'Running MIC test for {region} data', dynamic_ncols=True,
                               leave=False):
                mine = MINE()
                mine.compute_score(data[column].values.ravel(), self._data.get_labels()[region].values.ravel())
                score = mine.mic()

                if score >= correlation_threshold:
                    uncorrelated[region].remove(column)

            print(colored(f'\rApplied MIC test on selected features for {region}, {len(uncorrelated[region])} features are found', 'green'))

        return uncorrelated

    def apply_mic(self, correlation_threshold: float = 0.02) -> Dict[str, set]:
        uncorrelated = {
            region: set() for region in self._data.get_epigenomic_data().keys()
        }

        for region, data in self._data.get_epigenomic_data().items():
            if region == DataRetrieval.KEY_ENHANCERS:
                correlation_threshold = 0.01
            for column in tqdm(data.columns, desc=f'Running MIC test for {region} data', dynamic_ncols=True,
                               leave=False):
                mine = MINE()
                mine.compute_score(data[column].values.ravel(), self._data.get_labels()[region].values.ravel())
                score = mine.mic()

                if score < correlation_threshold:
                    uncorrelated[region].add(column)

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
                    total=len(data.columns), desc=f"Running Pearson test for {region} "
                                                  f"to find feature-feature correlations", dynamic_ncols=True,
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
                colored(f'\rApplied Pearson for feature-feature correlation for {region},'
                        f' {len(extremely_correlated[region])} useless features are found', 'green'))

        return extremely_correlated, scores

    def apply_spearman_for_features_correlation(self, spearman_threshold: float = 0.01,
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
                    total=len(data.columns), desc=f"Running Spearman test for {region} "
                                                  f"to find feature-feature correlations", dynamic_ncols=True,
                    leave=False):
                for feature in data.columns[i + 1:]:
                    correlation, p_value = spearmanr(data[column].values.ravel(), data[feature].values.ravel())
                    correlation = numpy.abs(correlation)
                    scores[region].append((correlation, column, feature))
                    if p_value < spearman_threshold and correlation > correlation_threshold:
                        print(region, column, feature, correlation)
                        if entropy(data[column]) > entropy(data[feature]):
                            extremely_correlated[region].add(feature)
                        else:
                            extremely_correlated[region].add(column)

            print(
                colored(f'\rApplied Spearman for feature-feature correlation for {region},'
                        f' {len(extremely_correlated[region])} useless features are found', 'green'))

        return extremely_correlated, scores

    def apply_boruta(self, max_iter: int = 10, threshold: float = 0.05, max_depth: int = 5) -> Dict[str, set]:
        features_to_drop = {
            region: set() for region in [DataRetrieval.KEY_PROMOTERS, DataRetrieval.KEY_ENHANCERS]
        }
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
            boruta_selector.fit(data.values, self._data.get_labels()[region].values.ravel())
            features_to_drop[region] = {data.columns[i] for i in range(len(boruta_selector.support_)) if
                                        boruta_selector.support_[i] == False}

        return features_to_drop
