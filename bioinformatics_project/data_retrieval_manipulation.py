from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome


def download_epigenomic_data(cell_line='HEK293', window_size=200):
    promoters_epigenomic_data, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )
    promoters_data = {'data': promoters_epigenomic_data, 'labels': promoters_labels}
    print('Data obtained: promoters epigenomes and labels')

    enhancers_epigenomic_data, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )
    enhancers_data = {'data': enhancers_epigenomic_data, 'labels': enhancers_labels}

    print('Data obtained: enhancers epigenomes and labels')

    return promoters_data, enhancers_data


def download_sequence_data(assembly='hg19'):
    hg19 = Genome(assembly)
    print('Data obtained: genome hg19 - ' + str(hg19))
    return hg19
