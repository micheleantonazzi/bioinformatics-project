import os

from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class DataPreprocessingPipeline:
    FOLDER_V0: str = 'preprocessed_data_v0'
    FOLDER_V1: str = 'preprocessed_data_v1'
    FOLDER_V2: str = 'preprocessed_data_v2'

    def __init__(self, data: DataRetrieval):
        self._data = data

    def execute_v0(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V0):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V0)
        else:
            print(colored('Data have not been already preprocessed, starting preprocess procedure', 'red'))
            promoters_features_number = len(self._data.get_promoters_epigenomic_data().columns)
            enhancers_features_number = len(self._data.get_enhancers_epigenomic_data().columns)
            data_preprocessing = DataPreprocessing(self._data)
            data_preprocessing.fill_nan_data()
            data_preprocessing.drop_constant_features()
            data_preprocessing.apply_z_scoring()
            self._data.save_epigenomic_data_to_csv(DataPreprocessingPipeline.FOLDER_V0)
            print(colored(f'The initial amount of features for promoters was {promoters_features_number}, '
                          f'after preprocessing it is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The initial amount of features for enhancers was {enhancers_features_number}, '
                          f'after preprocessing it is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))

        print(colored('Applied preprocessing version 0', 'red'))
        self._data.set_data_version('v0')
        return self._data

    def execute_v1(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V1):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V1)
        else:
            print(colored('Data have not been already preprocessed, starting preprocess procedure', 'red'))
            promoters_features_number = len(self._data.get_promoters_epigenomic_data().columns)
            enhancers_features_number = len(self._data.get_enhancers_epigenomic_data().columns)
            data_preprocessing = DataPreprocessing(self._data)
            data_preprocessing.fill_nan_data()
            data_preprocessing.drop_constant_features()
            data_preprocessing.apply_z_scoring()
            uncorrelated = data_preprocessing.apply_pearson_spearman_correlation()
            self._data.remove_uncorrelated_features(uncorrelated)
            uncorrelated = data_preprocessing.apply_mic()
            self._data.remove_uncorrelated_features(uncorrelated)
            uncorrelated, _ = data_preprocessing.apply_pearson_for_features_correlation()
            self._data.remove_uncorrelated_features(uncorrelated)
            uncorrelated = data_preprocessing.apply_boruta(200)
            self._data.remove_uncorrelated_features(uncorrelated)
            self._data.save_epigenomic_data_to_csv(DataPreprocessingPipeline.FOLDER_V1)
            print(colored(f'The initial amount of features for promoters was {promoters_features_number}, '
                          f'after preprocessing it is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The initial amount of features for enhancers was {enhancers_features_number}, '
                          f'after preprocessing it is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))

        print(colored('Applied preprocessing version 1', 'red'))
        self._data.set_data_version('v1')
        return self._data

    def execute_v2(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V2):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V2)
        else:
            print(colored('Data have not been already preprocessed, starting preprocess procedure', 'red'))
            promoters_features_number = len(self._data.get_promoters_epigenomic_data().columns)
            enhancers_features_number = len(self._data.get_enhancers_epigenomic_data().columns)
            data_preprocessing = DataPreprocessing(self._data)
            data_preprocessing.fill_nan_data()
            data_preprocessing.drop_constant_features()
            data_preprocessing.apply_z_scoring()
            uncorrelated = data_preprocessing.apply_pearson_spearman_correlation()
            uncorrelated = data_preprocessing.apply_mic_on_selected_features(uncorrelated)
            self._data.remove_uncorrelated_features(uncorrelated)
            uncorrelated, _ = data_preprocessing.apply_pearson_for_features_correlation()
            self._data.remove_uncorrelated_features(uncorrelated)
            uncorrelated = data_preprocessing.apply_boruta(300)
            self._data.remove_uncorrelated_features(uncorrelated)
            self._data.save_epigenomic_data_to_csv(DataPreprocessingPipeline.FOLDER_V2)
            print(colored(f'The initial amount of features for promoters was {promoters_features_number}, '
                          f'after preprocessing it is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The initial amount of features for enhancers was {enhancers_features_number}, '
                          f'after preprocessing it is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))

        print(colored('Applied preprocessing version 2', 'red'))
        self._data.set_data_version('v2')
        return self._data

