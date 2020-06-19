from tqdm import tqdm

from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing_pipeline import DataPreprocessingPipeline
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.model_builder import ModelBuilder
from bioinformatics_project.models.parameter_selector import ParameterSelector
from sklearn.model_selection import StratifiedShuffleSplit


class ExperimentExecutor:
    def get_holdouts(self, splits):
        return StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

    def execute_promoters_epigenomic_experiment(self, splits: int = 50):
        data_retrieval = DataRetrieval()
        DataPreprocessingPipeline(data_retrieval).execute_v2()

        holdouts = self.get_holdouts(splits)

        data, labels = data_retrieval.get_epigenomic_data_for_learning()[DataRetrieval.KEY_PROMOTERS]
        parameters_function = ParameterSelector(data_retrieval).get_functions()

        for i, (train, test) in tqdm(enumerate(holdouts.split(data, labels)), total=splits,
                                     desc="Computing holdouts", dynamic_ncols=True):

            models = ModelBuilder(data_retrieval).get_functions()
            for model_name, builder in tqdm(models.items(),
                                                 total=len(models), desc="Training models", leave=False, dynamic_ncols=True):

                print(f'For {DataRetrieval.KEY_PROMOTERS} train {model_name}')
                model, train_parameters = builder(DataRetrieval.KEY_PROMOTERS, parameters_function[model_name]()[DataRetrieval.KEY_PROMOTERS])
                model.fit(data[train], labels[train], **train_parameters)

