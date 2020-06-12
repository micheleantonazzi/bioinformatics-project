from typing import Dict

from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
import pandas
import numpy
from keras_bed_sequence import BedSequence
from termcolor import colored


class DataRetrieval:
    def __init__(self, cell_line: str = 'HEK293', window_size: int = 200):
        self.key_epigenomic: str = 'epigenomic_data'
        self.key_sequence: str = 'sequence_data'
        self.key_labels = 'labels'
        self._cell_line: str = cell_line
        self._window_size: int = window_size
        self._promoters_data: Dict[str, pandas.DataFrame] = {self.key_epigenomic: None, self.key_sequence: None,
                                                             self.key_labels: None}
        self._enhancers_data: Dict[str, pandas.DataFrame] = {self.key_epigenomic: None, self.key_sequence: None,
                                                             self.key_labels: None}
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
        self._promoters_data[self.key_epigenomic] = promoters_epigenomic_data
        self._promoters_data[self.key_labels] = promoters_labels

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
        self._enhancers_data[self.key_epigenomic] = promoters_enhancers_data
        self._enhancers_data[self.key_labels] = enhancers_labels

        print(colored('Data obtained: enhancers epigenomic data and labels', 'green'))

        return self._enhancers_data

    def load_genome_data(self, assembly: str = 'hg19') -> Genome:
        print('Starting loading genome sequence of ' + assembly)
        self._genome = Genome(assembly)
        print(colored('\rData obtained: genome sequence of ' + str(self._genome), 'green'))
        return self._genome

    def extract_promoters_sequence_data(self, quantity: int = -1) -> pandas.DataFrame:
        print('Starting extracting promoters sequence data')

        if self._promoters_data[self.key_epigenomic] is not None and quantity == -1:
            quantity = len(self._promoters_data[self.key_epigenomic])

        one_not_encode = numpy.array(BedSequence(
            self._genome,
            bed=self._promoters_data[self.key_epigenomic].reset_index()[
                    self._promoters_data[self.key_epigenomic].index.names][:quantity],
            nucleotides='actg',
            batch_size=1
        ))

        self._promoters_data[self.key_sequence] = pandas.DataFrame(
            one_not_encode.reshape(-1, self._window_size * 4).astype(int),
            columns=[f"{i}{nucleotide}" for i in range(self._window_size) for nucleotide in 'actg']
        )

        print(colored('\rData loading: promoters sequence data', 'green'))
        return self._promoters_data[self.key_sequence]

    def extract_enhancers_sequence_data(self, quantity: int = -1) -> pandas.DataFrame:
        print('Starting extracting enhancers sequence data')

        if self._enhancers_data[self.key_epigenomic] is not None and quantity == -1:
            quantity = len(self._enhancers_data[self.key_epigenomic])

        one_not_encode = numpy.array(BedSequence(
            self._genome,
            bed=self._enhancers_data[self.key_epigenomic].reset_index()[
                    self._enhancers_data[self.key_epigenomic].index.names][:quantity],
            nucleotides='actg',
            batch_size=1
        ))

        self._enhancers_data[self.key_sequence] = pandas.DataFrame(
            one_not_encode.reshape(-1, self._window_size * 4).astype(int),
            columns=[f"{i}{nucleotide}" for i in range(self._window_size) for nucleotide in 'actg']
        )

        print(colored('\rData loading: enhancers sequence data', 'green'))
        return self._enhancers_data[self.key_sequence]

    def set_promoters_epigenomic_data(self, new_data: pandas.DataFrame):
        self._promoters_data[self.key_epigenomic] = new_data

    def get_promoters_data(self) -> Dict[str, pandas.DataFrame]:
        return self._promoters_data

    def get_promoters_epigenomic_data(self) -> pandas.DataFrame:
        return self._promoters_data[self.key_epigenomic]

    def get_promoters_labels(self) -> pandas.DataFrame:
        return self._promoters_data[self.key_labels]

    def get_enhancers_data(self) -> Dict[str, pandas.DataFrame]:
        return self._enhancers_data

    def get_enhancers_epigenomic_data(self) -> pandas.DataFrame:
        return self._enhancers_data[self.key_epigenomic]

    def get_enhancers_labels(self) -> pandas.DataFrame:
        return self._enhancers_data[self.key_labels]

    def get_genome_data(self) -> Genome:
        return self._genome
