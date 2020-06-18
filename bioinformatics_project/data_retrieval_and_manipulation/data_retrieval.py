import os
from typing import Dict

from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
import pandas
import numpy
from keras_bed_sequence import BedSequence
from termcolor import colored


class DataRetrieval:
    KEY_EPIGENOMIC: str = 'epigenomic_data'
    KEY_SEQUENCE: str = 'sequence_data'
    KEY_PROMOTERS: str = 'promoters'
    KEY_ENHANCERS: str = 'enhancers'
    KEY_LABELS: str = 'labels'

    def __init__(self, cell_line: str = 'HEK293', window_size: int = 200):
        self._cell_line: str = cell_line
        self._window_size: int = window_size
        self._promoters_data: Dict[str, pandas.DataFrame] = {DataRetrieval.KEY_EPIGENOMIC: None,
                                                             DataRetrieval.KEY_SEQUENCE: None,
                                                             DataRetrieval.KEY_LABELS: None}
        self._enhancers_data: Dict[str, pandas.DataFrame] = {DataRetrieval.KEY_EPIGENOMIC: None,
                                                             DataRetrieval.KEY_SEQUENCE: None,
                                                             DataRetrieval.KEY_LABELS: None}
        self._genome: Genome = None

    def load_promoters_epigenomic_data(self) -> Dict[str, pandas.DataFrame]:
        print(
            'Starting loading promoters epigenomic data of ' + self._cell_line +
            ' cell line with window size of ' + str(self._window_size))

        promoters_epigenomic_data, promoters_labels = load_epigenomes(
            cell_line=self._cell_line,
            dataset="fantom",
            regions="promoters",
            window_size=self._window_size
        )
        self._promoters_data[DataRetrieval.KEY_EPIGENOMIC] = promoters_epigenomic_data
        self._promoters_data[DataRetrieval.KEY_LABELS] = promoters_labels

        print(colored('Data obtained: promoters epigenomic data and labels', 'green'))

        return self._promoters_data

    def load_enhancers_epigenomic_data(self) -> Dict[str, pandas.DataFrame]:
        print(
            'Starting loading enhancers epigenomic data of ' + self._cell_line +
            ' cell line with window size of ' + str(self._window_size))

        promoters_enhancers_data, enhancers_labels = load_epigenomes(
            cell_line=self._cell_line,
            dataset='fantom',
            regions='enhancers',
            window_size=self._window_size
        )
        self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC] = promoters_enhancers_data
        self._enhancers_data[DataRetrieval.KEY_LABELS] = enhancers_labels

        print(colored('Data obtained: enhancers epigenomic data and labels', 'green'))

        return self._enhancers_data

    def load_genome_data(self, assembly: str = 'hg19') -> Genome:
        print('Starting loading genome sequence of ' + assembly)
        self._genome = Genome(assembly)
        print(colored('\rData obtained: genome sequence of ' + str(self._genome), 'green'))
        return self._genome

    def extract_promoters_sequence_data(self, quantity: int = -1) -> pandas.DataFrame:
        print('Starting extracting promoters sequence data')

        if self._promoters_data[DataRetrieval.KEY_EPIGENOMIC] is not None and quantity == -1:
            quantity = len(self._promoters_data[DataRetrieval.KEY_EPIGENOMIC])

        one_not_encode = numpy.array(BedSequence(
            self._genome,
            bed=self._promoters_data[DataRetrieval.KEY_EPIGENOMIC].reset_index()[
                    self._promoters_data[DataRetrieval.KEY_EPIGENOMIC].index.names][:quantity],
            nucleotides='actg',
            batch_size=1
        ))

        self._promoters_data[DataRetrieval.KEY_SEQUENCE] = pandas.DataFrame(
            one_not_encode.reshape(-1, self._window_size * 4).astype(int),
            columns=[f"{i}{nucleotide}" for i in range(self._window_size) for nucleotide in 'actg']
        )

        print(colored('\rData loading: promoters sequence data', 'green'))
        return self._promoters_data[DataRetrieval.KEY_SEQUENCE]

    def extract_enhancers_sequence_data(self, quantity: int = -1) -> pandas.DataFrame:
        print('Starting extracting enhancers sequence data')

        if self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC] is not None and quantity == -1:
            quantity = len(self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC])

        one_not_encode = numpy.array(BedSequence(
            self._genome,
            bed=self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC].reset_index()[
                    self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC].index.names][:quantity],
            nucleotides='actg',
            batch_size=1
        ))

        self._enhancers_data[DataRetrieval.KEY_SEQUENCE] = pandas.DataFrame(
            one_not_encode.reshape(-1, self._window_size * 4).astype(int),
            columns=[f"{i}{nucleotide}" for i in range(self._window_size) for nucleotide in 'actg']
        )

        print(colored('\rData loading: enhancers sequence data', 'green'))
        return self._enhancers_data[DataRetrieval.KEY_SEQUENCE]

    def remove_uncorrelated_features(self, uncorrelated: Dict[str, set]):
        self._promoters_data[DataRetrieval.KEY_EPIGENOMIC] = self._promoters_data[DataRetrieval.KEY_EPIGENOMIC].drop(
            columns=[col for col in
                     uncorrelated[DataRetrieval.KEY_PROMOTERS]
                     if col in self._promoters_data[DataRetrieval.KEY_EPIGENOMIC]]
        )

        self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC] = self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC].drop(
            columns=[col for col in
                     uncorrelated[DataRetrieval.KEY_ENHANCERS]
                     if col in self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC]]
        )

    def set_promoters_epigenomic_data(self, new_data: pandas.DataFrame):
        self._promoters_data[DataRetrieval.KEY_EPIGENOMIC] = new_data

    def set_enhancers_epigenomic_data(self, new_data: pandas.DataFrame):
        self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC] = new_data

    def set_epigenomic_data(self, region: str, data: pandas.DataFrame):
        if region == DataRetrieval.KEY_PROMOTERS:
            self._promoters_data[DataRetrieval.KEY_EPIGENOMIC] = data
        elif region == DataRetrieval.KEY_ENHANCERS:
            self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC] = data

    def get_promoters_data(self) -> Dict[str, pandas.DataFrame]:
        return self._promoters_data

    def get_promoters_epigenomic_data(self) -> pandas.DataFrame:
        return self._promoters_data[DataRetrieval.KEY_EPIGENOMIC]

    def get_promoters_labels(self) -> pandas.DataFrame:
        return self._promoters_data[DataRetrieval.KEY_LABELS]

    def get_enhancers_data(self) -> Dict[str, pandas.DataFrame]:
        return self._enhancers_data

    def get_enhancers_epigenomic_data(self) -> pandas.DataFrame:
        return self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC]

    def get_enhancers_labels(self) -> pandas.DataFrame:
        return self._enhancers_data[DataRetrieval.KEY_LABELS]

    def get_genome_data(self) -> Genome:
        return self._genome

    def get_epigenomic_data(self) -> Dict[str, pandas.DataFrame]:
        return {DataRetrieval.KEY_PROMOTERS: self._promoters_data[DataRetrieval.KEY_EPIGENOMIC],
                DataRetrieval.KEY_ENHANCERS: self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC]}

    def get_labels(self) -> Dict[str, pandas.DataFrame]:
        return {DataRetrieval.KEY_PROMOTERS: self._promoters_data[DataRetrieval.KEY_LABELS],
                DataRetrieval.KEY_ENHANCERS: self._enhancers_data[DataRetrieval.KEY_LABELS]}

    def get_sequence_data(self) -> Dict[str, pandas.DataFrame]:
        return {DataRetrieval.KEY_PROMOTERS: self._promoters_data[DataRetrieval.KEY_SEQUENCE],
                DataRetrieval.KEY_ENHANCERS: self._enhancers_data[DataRetrieval.KEY_SEQUENCE]}

    def get_epigenomic_data_for_learning(self):
        return {region: (self.get_epigenomic_data()[region].values, self.get_labels()[region].values.ravel())
                for region in [DataRetrieval.KEY_PROMOTERS, DataRetrieval.KEY_ENHANCERS]}

    def check_exists_data_preprocessed(self, folder) -> bool:
        return os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder,
                                           'promoters_epigenomic_data_processed.csv.gz')) and os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), folder,
                         'enhancers_epigenomic_data_processed.csv.gz'))

    def save_epigenomic_data_to_csv(self, folder):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder))
        self._promoters_data[DataRetrieval.KEY_EPIGENOMIC].to_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), folder,
                         'promoters_epigenomic_data_processed.csv.gz'))
        self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC].to_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), folder,
                         'enhancers_epigenomic_data_processed.csv.gz'))

    def load_epigenomic_data_from_csv(self, folder: str):
        self._promoters_data[DataRetrieval.KEY_EPIGENOMIC] = pandas.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), folder,
                         'promoters_epigenomic_data_processed.csv.gz'),
            index_col=['chrom', 'chromStart', 'chromEnd', 'strand'])
        self._enhancers_data[DataRetrieval.KEY_EPIGENOMIC] = pandas.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), folder,
                         'enhancers_epigenomic_data_processed.csv.gz'),
            index_col=['chrom', 'chromStart', 'chromEnd', 'strand'])
        print(colored('Epigenomic data saved to csv', 'green'))
