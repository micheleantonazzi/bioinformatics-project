from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
import pandas
import numpy
from keras_bed_sequence import BedSequence


def download_epigenomic_data(cell_line='HEK293', window_size=200):
    promoters_epigenomic_data, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )
    promoters_data = {'data': promoters_epigenomic_data, 'labels': promoters_labels}

    enhancers_epigenomic_data, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )
    enhancers_data = {'data': enhancers_epigenomic_data, 'labels': enhancers_labels}

    return promoters_data, enhancers_data


def download_genome_data(assembly='hg19'):
    hg19 = Genome(assembly)
    return hg19


def extract_sequence_data(genome: Genome, data: pandas.DataFrame):
    res = genome.bed_to_sequence(data.reset_index()[data.index.names])
    return res


def sequence_data_flat_one_not_encoded(genome: Genome, data: pandas.DataFrame, window_size: int,
                                       nucleotides: str = 'actg') -> numpy.ndarray:
    one_not_encode = numpy.array(BedSequence(
        genome,
        bed=data.reset_index()[data.index.names],
        nucleotides=nucleotides,
        batch_size=1
    ))

    return one_not_encode.reshape(-1, window_size * 4).astype(int)


def sequence_data_to_dataframe(array: numpy.ndarray, window_size: int, nucleotides: str = 'actg') -> pandas.DataFrame:
    return pandas.DataFrame(array,
                            columns=[f"{i}{nucleotide}" for i in range(window_size) for nucleotide in nucleotides])
