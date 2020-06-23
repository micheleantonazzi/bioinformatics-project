import os
import pickle

import pandas
from tqdm import tqdm

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.model_builder_epigenomic import ModelBuilderEpigenomic
from bioinformatics_project.models.parameter_selector_epigenomic import ParameterSelectorEpigenomic
from sklearn.model_selection import StratifiedShuffleSplit
import numpy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sanitize_ml_labels import sanitize_ml_labels
from barplots import barplots


class ExperimentExecutor:
    def get_holdouts(self, splits):
        return StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

    def calculate_metrics(self, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> numpy.ndarray:
        integer_metrics = accuracy_score, balanced_accuracy_score
        float_metrics = roc_auc_score, average_precision_score
        results1 = {
            sanitize_ml_labels(metric.__name__): metric(y_true, numpy.round(y_pred))
            for metric in integer_metrics
        }
        results2 = {
            sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
            for metric in float_metrics
        }
        return {
            **results1,
            **results2
        }

    def save_results(self, data_version: str, region: str, model_name: str, results: list):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'epigenomic', 'results_' + data_version, region)
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, model_name + '.pkl')
        with open(os.path.join(path), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    def load_results(self, data_version: str, region: str, model_name: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'epigenomic', 'results_' + data_version, region, model_name + '.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        return []

    def print_results(self, experiment_type: str, data_version: str, region: str, results: pandas.DataFrame):
        results = results.drop(columns=['holdout', 'region'])
        height = len(results['model'].unique())
        barplots(
            results,
            groupby=["model", "run_type"],
            show_legend=False,
            height=height,
            orientation="horizontal",
            path='experiment/' + experiment_type + '/plots_' + data_version + '/' + region + '/{feature}.png'
        )

    def execute_epigenomic_experiment(self, data_retrieval: DataRetrieval, region: str, splits: int = 50):

        holdouts = self.get_holdouts(splits)

        data, labels = data_retrieval.get_epigenomic_data_for_learning()[region]
        parameters_function = ParameterSelectorEpigenomic(data_retrieval).get_epigenomic_functions()
        models = ModelBuilderEpigenomic(data_retrieval).get_epigenomic_functions()

        results = []
        for model_name, builder in tqdm(models.items(),
                                        total=len(models), desc="Training models", leave=False, dynamic_ncols=True):

            model_results = self.load_results(data_retrieval.get_data_version(), region, model_name)

            if len(model_results) < splits * 2:
                for i, (train, test) in tqdm(enumerate(holdouts.split(data, labels)), total=splits,
                                             desc="Computing holdouts", dynamic_ncols=True):
                    print(f'For {region} train {model_name}')
                    model, train_parameters = builder(region,
                                                      parameters_function[model_name]()[region])

                    model.fit(data[train], labels[train], **train_parameters)

                    model_results.append({
                        'region': region,
                        'model': model_name,
                        'run_type': 'train',
                        'holdout': i,
                        **self.calculate_metrics(labels[train], model.predict(data[train]))
                    })
                    model_results.append({
                        'region': region,
                        'model': model_name,
                        'run_type': 'test',
                        'holdout': i,
                        **self.calculate_metrics(labels[test], model.predict(data[test]))
                    })
                self.save_results(data_retrieval.get_data_version(), region, model_name, model_results)

            results = results + model_results

        results = pandas.DataFrame(results)
        self.print_results('epigenomic', data_retrieval.get_data_version(), region, results)
        return results
