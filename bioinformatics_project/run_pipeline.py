from data_retrieval_manipulation import *

if __name__ == '__main__':
    window_size = 200
    cell_line = 'HEK293'

    promoters_data, enhancers_data = download_epigenomic_data(cell_line, window_size)
    print('Data obtained: promoters and enhancers epigenomes and labels')

    hg19 = download_genome_data()
    print('Data obtained: genome hg19 - ' + str(hg19))

    promoters_sequence_data = extract_sequence_data(hg19, promoters_data["data"])
    print("Data obtained: promoters sequence data")

    enhancers_sequence_data = extract_sequence_data(hg19, enhancers_data["data"])
    print("Data obtained: enhancers sequence data")

    promoters_sequence_one_not = sequence_data_flat_one_not_encoded(hg19, promoters_data['data'], window_size, 'actg')
    print(len(promoters_sequence_one_not))

    enhancers_sequence_one_not = sequence_data_flat_one_not_encoded(hg19, enhancers_data['data'], window_size, 'actg')
    print(len(enhancers_sequence_one_not))
