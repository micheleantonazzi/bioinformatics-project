import os

from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval

"""
    This class contains methods to automatically execute the pipelines to preprocess data.
    Once a pipeline is run, the dataset is saved to disk in order to load it in future.  
"""
class DataPreprocessingPipeline:
    FOLDER_V0: str = 'preprocessed_data_v0'
    FOLDER_V1: str = 'preprocessed_data_v1'
    FOLDER_V2: str = 'preprocessed_data_v2'
    FOLDER_V3: str = 'preprocessed_data_v3'


    def __init__(self, data: DataRetrieval):
        self._data = data

    """
        Execute the easier preprocessing pipeline, without feature selection. This pipeline is to fast to execute and
        it isn't saved on disk. 
            - fill nan values 
            - drop constant feature (there are not constant feature)
            - apply z-score
    """
    def execute_v0(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V0):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V0)
            print(colored(f'The amount of features for promoters is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The amount of features for enhancers is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))
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

    """
        The first version of pipeline with feature selection.
        This pipeline is wrong because the uncorrelated feature found using Pearson and Spearman are directly removed,
        without check if they have non-linear correlation using MIC test.
            - fill nan values 
            - drop constant feature (there are not constant feature)
            - apply z-score
            - execute Pearson and Spearman tests
            - remove feature found with the previous step
            - apply MIC on all dataset
            - remove feature found with the previous step
            - apply Pearson test to feature-feature correlation
            - remove feature with less entropy
            - run Boruta
            - remove feature found with the previous step

    """
    def execute_v1(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V1):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V1)
            print(colored(f'The amount of features for promoters is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The amount of features for enhancers is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))
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

    """
        PIPELINE USED IN THE EXPERIMENTS
        
        The fixed version of pipeline with feature selection.
            - fill nan values 
            - drop constant feature (there are not constant feature)
            - apply z-score
            - execute Pearson and Spearman tests
            - apply MIC on feature found in the previous step to find non-linear correlation
            - remove feature left over from the previous step
            - apply Pearson test to feature-feature correlation
            - remove feature with less entropy
            - run Boruta
            - remove feature found with the previous step
    """
    def execute_v2(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V2):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V2)
            print(colored(f'The amount of features for promoters is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The amount of features for enhancers is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))
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

    """        
        This pipeline is similar to the v2, the only difference is that with are remove only the feature rejected from Boruta
        and those tentative are confirmed
            - fill nan values 
            - drop constant feature (there are not constant feature)
            - apply z-score
            - execute Pearson and Spearman tests
            - apply MIC on feature found in the previous step to find non-linear correlation
            - remove feature left over from the previous step
            - apply Pearson test to feature-feature correlation
            - remove feature with less entropy
            - run Boruta without remove tentative features
            - remove feature found with the previous step
    """
    def execute_v3(self) -> DataRetrieval:
        self._data.load_promoters_epigenomic_data()
        self._data.load_enhancers_epigenomic_data()

        if self._data.check_exists_data_preprocessed(DataPreprocessingPipeline.FOLDER_V3):
            print(colored('Data have been already preprocessed', 'green'))
            self._data.load_epigenomic_data_from_csv(DataPreprocessingPipeline.FOLDER_V3)
            print(colored(f'The amount of features for promoters is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The amount of features for enhancers is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))
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
            uncorrelated = data_preprocessing.apply_boruta_without_drop_tentative(200)
            self._data.remove_uncorrelated_features(uncorrelated)
            self._data.save_epigenomic_data_to_csv(DataPreprocessingPipeline.FOLDER_V3)
            print(colored(f'The initial amount of features for promoters was {promoters_features_number}, '
                          f'after preprocessing it is {len(self._data.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The initial amount of features for enhancers was {enhancers_features_number}, '
                          f'after preprocessing it is {len(self._data.get_enhancers_epigenomic_data().columns)}', 'green'))

        print(colored('Applied preprocessing version 3', 'red'))
        self._data.set_data_version('v3')
        return self._data

