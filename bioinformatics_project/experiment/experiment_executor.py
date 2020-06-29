import os
import pickle
from typing import Tuple

import pandas
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from tabulate import tabulate
from tqdm import tqdm
from ucsc_genomes_downloader import Genome

from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.model_builder_epigenomic import ModelBuilderEpigenomic
from bioinformatics_project.models.model_builder_sequence import ModelBuilderSequence
from bioinformatics_project.models.parameter_selector_epigenomic import ParameterSelectorEpigenomic
from sklearn.model_selection import StratifiedShuffleSplit
import numpy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sanitize_ml_labels import sanitize_ml_labels
from barplots import barplots
from tensorflow.keras.utils import Sequence

from bioinformatics_project.models.parameter_selector_sequence import ParameterSelectorSequence


"""
    This class contains the procedures to execute both epigenomic and sequence experiments.
    After the procedures is done, the results are saved to disk and the graphs are plotted.
"""
class ExperimentExecutor:
    def get_holdouts(self, splits):
        return StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

    """
        Return the sequence holdout specified by parameters.
        In particular, it is returned a tuple containing two MixedSequence that load seqauence data for both training and test sets
    """
    def get_sequence_holdout(self, train: numpy.ndarray, test: numpy.ndarray, bed: pandas.DataFrame, labels: numpy.ndarray, genome: Genome, batch_size: int = 1024) -> Tuple[Sequence, Sequence]:
        return (
            MixedSequence(
                x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
                y=labels[train],
                batch_size=batch_size
            ),
            MixedSequence(
                x=BedSequence(genome, bed.iloc[test], batch_size=batch_size),
                y=labels[test],
                batch_size=batch_size
            )
        )

    """
        Calculate metrics relative to an experiments. The input parameters are the correct labels and the results  given by a model
    """
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

    """
        Save results to disk in order to not recalculate it in future.
        Results are saved in a path based on the type of experiment (which uses epigenomic or sequence data),
        the version of the preprocessing pipeline, the region (promoters or enhancers) and the file name corresponds
        to the model name
    """
    def save_results(self, experiment_type: str, data_version: str, region: str, model_name: str, results: list):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), experiment_type, 'results_' + data_version, region)
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, model_name + '.pkl')
        with open(os.path.join(path), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    """
        Return the results specific for experiment type, version of data, region and model.
        They are loaded from disk if the have been already calculated, otherwise an empty list is returned
    """
    def load_results(self, experiment_type: str, data_version: str, region: str, model_name: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), experiment_type, 'results_' + data_version, region, model_name + '.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        return []

    """
        Plot the graphs of the results in a specific folder based on experiment type, data version, region and model name.
        Write the markdown tables of the metric results in a file
    """
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

        open('experiment/' + experiment_type + '/plots_' + data_version + '/' + region + '/metrics_table.txt', 'w').close()
        file = open('experiment/' + experiment_type + '/plots_' + data_version + '/' + region + '/metrics_table.txt', 'w')
        models = results.model.unique()
        run_types = results.run_type.unique()
        for metric in ['Accuracy', 'AUROC', 'AUPRC']:
            temp = {run_type: [] for run_type in run_types}
            for model in models:
                for run_type in run_types:
                    res = results[(results['model'] == model) & (results['run_type'] == run_type)][metric].values
                    temp[run_type].append(f'mean = {round(numpy.mean(res), 4)}\nSTD = {round(numpy.std(res), 4)}')

            df = pandas.DataFrame({
                'Models': models,
                'Training': temp['train'],
                'Test': temp['test']
            }).set_index('Models')
            file.writelines(f'Table for {region} {experiment_type} experiment, metric {metric}\n')
            file.writelines(tabulate(df, tablefmt="pipe", headers="keys") + '\n\n')
        file.close()

    """
        Execute the experiments using epigenomic data.
        All models are runned for each holdout, the result are saved to disk.
        If the results for a specific model have been already calculated, they are loaded directly from disk.
        Finally the metrics are calculated and the relative graphs are plotted
    """
    def execute_epigenomic_experiment(self, data_retrieval: DataRetrieval, region: str, splits: int = 50):

        holdouts = self.get_holdouts(splits)

        data, labels = data_retrieval.get_epigenomic_data_for_learning()[region]
        parameters_function = ParameterSelectorEpigenomic(data_retrieval).get_epigenomic_functions()
        models = ModelBuilderEpigenomic(data_retrieval).get_epigenomic_functions()

        results = []
        for model_name, builder in tqdm(models.items(),
                                        total=len(models), desc="Training models", leave=False, dynamic_ncols=True):

            model_results = self.load_results('epigenomic', data_retrieval.get_data_version(), region, model_name)

            if len(model_results) < splits * 2:
                for i, (train, test) in tqdm(enumerate(holdouts.split(data, labels)), total=splits,
                                             desc="Computing holdouts", dynamic_ncols=True):
                    print(f'For {region} train {model_name}')
                    model, train_parameters = builder(region,
                                                      parameters_function[model_name](labels[train])[region], labels[train])

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
                self.save_results('epigenomic', data_retrieval.get_data_version(), region, model_name, model_results)

            results = results + model_results

        results = pandas.DataFrame(results)
        self.print_results('epigenomic', data_retrieval.get_data_version(), region, results)
        return results

    """
        Execute the experiments using sequence data.
        All models are runned for each holdout, the result are saved to disk.
        If the results for a specific model have been already calculated, they are loaded directly from disk.
        Finally the metrics are calculated and the relative graphs are plotted
    """
    def execute_sequence_experiment(self, data_retrieval: DataRetrieval, region: str, splits: int = 3):
        data_retrieval.load_genome_data()
        holdouts = self.get_holdouts(splits)

        parameters_functions = ParameterSelectorSequence(data_retrieval).get_sequence_functions()
        bed, labels = data_retrieval.get_sequence_data_for_learning()[region]
        models = ModelBuilderSequence(data_retrieval).get_sequence_functions()

        results = []
        for i, (train_index, test_index) in tqdm(enumerate(holdouts.split(bed, labels)), total=splits, desc="Computing holdouts", dynamic_ncols=True):
            train, test = self.get_sequence_holdout(train_index, test_index, bed, labels, data_retrieval.get_genome_data())

            for model_name, builder in tqdm(models.items(), total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
                model_results = list(filter(
                    lambda item: item['holdout'] == i,
                    self.load_results('sequence', data_retrieval.get_data_version(), region, model_name)
                ))
                if len(model_results) == 0:
                    model = builder()
                    history = model.fit(
                            train,
                            steps_per_epoch=train.steps_per_epoch,
                            validation_data=test,
                            validation_steps=test.steps_per_epoch,
                            **parameters_functions[model_name]()[region]
                        ).history
                    scores = pandas.DataFrame(history).iloc[-1].to_dict()
                    model_results.append({
                        'region': region,
                        "model": model_name,
                        "run_type": "train",
                        "holdout": i,
                        **{
                            sanitize_ml_labels(key): value
                            for key, value in scores.items()
                            if not key.startswith("val_")
                        }
                    })
                    model_results.append({
                        'region': region,
                        "model": model_name,
                        "run_type": "test",
                        "holdout": i,
                        **{
                            sanitize_ml_labels(key[4:]): value
                            for key, value in scores.items()
                            if key.startswith("val_")
                        }
                    })

                results = results + model_results
        for model_name, _ in models.items():
            self.save_results('sequence', data_retrieval.get_data_version(), region, model_name,
                              list(filter(lambda item: item['model'] == model_name, results)))

        results = pandas.DataFrame(results)
        self.print_results('sequence', data_retrieval.get_data_version(), region, results)

        return results
