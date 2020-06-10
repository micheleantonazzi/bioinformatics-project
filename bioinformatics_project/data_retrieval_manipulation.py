from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
import pandas


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
