from bioinformatics_project.data_retrieval_and_manipulation.data_preprocessing_pipeline import DataPreprocessingPipeline
from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval
from bioinformatics_project.models.model_builder import ModelBuilder
from bioinformatics_project.models.parameter_selector import ParameterSelector


class ExperimentExecutor:
    def execute_promoters_epigenomic_experiment(self):
        data_retrieval = DataRetrieval()
        DataPreprocessingPipeline(data_retrieval).execute_v2()

        parameters_function = ParameterSelector(data_retrieval).get_functions()

        for region, (data, label) in list(data_retrieval.get_epigenomic_data_for_learning().items())[:1]:
            for model, builder in ModelBuilder(data_retrieval).get_functions().items():
                print(f'For {region} train {model}')
                model, train_parameters = builder(region, parameters_function[model]()[region])
                model.fit(data, label, **train_parameters)
