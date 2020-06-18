import os

from termcolor import colored

from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing import DataPreprocessing
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval


class DataPreprocessingPipeline:
    def execute_v1(self):
        data_retrieval = DataRetrieval()
        data_retrieval.load_promoters_epigenomic_data()
        data_retrieval.load_enhancers_epigenomic_data()
        data_retrieval.load_genome_data()
        data_retrieval.extract_promoters_sequence_data()
        data_retrieval.extract_enhancers_sequence_data()

        if data_retrieval.check_exists_data_preprocessed():
            print(colored('Data have been already preprocessed', 'green'))
            data_retrieval.load_epigenomic_data_from_csv()
        else:
            print(colored('Data have not been already preprocessed, starting preprocess procedure', 'red'))
            promoters_features_number = len(data_retrieval.get_promoters_epigenomic_data().columns)
            enhancers_features_number = len(data_retrieval.get_enhancers_epigenomic_data().columns)
            data_preprocessing = DataPreprocessing(data_retrieval)
            data_preprocessing.fill_nan_data()
            data_preprocessing.drop_constant_features()
            data_preprocessing.apply_z_scoring()
            uncorrelated = data_preprocessing.apply_pearson_spearman_correlation()
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            uncorrelated = data_preprocessing.apply_mic()
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            uncorrelated, _ = data_preprocessing.apply_pearson_for_features_correlation()
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            uncorrelated = data_preprocessing.apply_boruta(200)
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            data_retrieval.save_epigenomic_data_to_csv()
            print(colored(f'The initial amount of features for promoters was {promoters_features_number}, '
                          f'after preprocessing it is {len(data_retrieval.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The initial amount of features for enhancers was {enhancers_features_number}, '
                          f'after preprocessing it is {len(data_retrieval.get_enhancers_epigenomic_data().columns)}', 'green'))

    def execute_v2(self):
        data_retrieval = DataRetrieval()
        data_retrieval.load_promoters_epigenomic_data()
        data_retrieval.load_enhancers_epigenomic_data()
        data_retrieval.load_genome_data()
        data_retrieval.extract_promoters_sequence_data()
        data_retrieval.extract_enhancers_sequence_data()

        if data_retrieval.check_exists_data_preprocessed():
            print(colored('Data have been already preprocessed', 'green'))
            data_retrieval.load_epigenomic_data_from_csv()
        else:
            print(colored('Data have not been already preprocessed, starting preprocess procedure', 'red'))
            promoters_features_number = len(data_retrieval.get_promoters_epigenomic_data().columns)
            enhancers_features_number = len(data_retrieval.get_enhancers_epigenomic_data().columns)
            data_preprocessing = DataPreprocessing(data_retrieval)
            data_preprocessing.fill_nan_data()
            data_preprocessing.drop_constant_features()
            data_preprocessing.apply_z_scoring()
            uncorrelated = data_preprocessing.apply_pearson_spearman_correlation()
            uncorrelated = data_preprocessing.apply_mic_on_selected_features(uncorrelated)
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            uncorrelated, _ = data_preprocessing.apply_pearson_for_features_correlation()
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            uncorrelated = data_preprocessing.apply_boruta(300)
            data_retrieval.remove_uncorrelated_features(uncorrelated)
            data_retrieval.save_epigenomic_data_to_csv()
            print(colored(f'The initial amount of features for promoters was {promoters_features_number}, '
                          f'after preprocessing it is {len(data_retrieval.get_promoters_epigenomic_data().columns)}', 'green'))
            print(colored(f'The initial amount of features for enhancers was {enhancers_features_number}, '
                          f'after preprocessing it is {len(data_retrieval.get_enhancers_epigenomic_data().columns)}', 'green'))


