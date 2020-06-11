from bioinformatics_project.data_retrieval_and_manipulation.data_retrieval import DataRetrieval

data_retrieval = DataRetrieval()
data_retrieval.load_promoters_epigenomic_data()
data_retrieval.load_enhancers_epigenomic_data()
genome = data_retrieval.load_genome_data()
data_retrieval.extract_promoters_sequence_data()
data_retrieval.extract_enhancers_sequence_data()
