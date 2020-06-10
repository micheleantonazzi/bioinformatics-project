from bioinformatics_project.data_retrieval_manipulation import *

if __name__ == '__main__':
    window_size = 200
    cell_line = 'HEK293'

    promoters_epigenomic_data, enhancers_epigenomic_data = download_epigenomic_data(cell_line, window_size)
    print('Data obtained: promoters and enhancers epigenomes and labels')

    hg19 = download_genome_data()
    print('Data obtained: genome hg19 - ' + str(hg19))

    #promoters_sequence_data = extract_sequence_data(hg19, promoters_epigenomic_data["data"])
    #print("Data obtained: promoters sequence data alphanumeric")

    #enhancers_sequence_data = extract_sequence_data(hg19, enhancers_epigenomic_data["data"])
    #print("Data obtained: enhancers sequence data alphanumeric")

    promoters_sequence_one_not = sequence_data_flat_one_not_encoded(hg19, promoters_epigenomic_data['data'], window_size, 'actg')
    promoters_sequence_data = sequence_data_to_dataframe(promoters_sequence_one_not, window_size, 'actg')
    print("Data obtained: promoters sequence data flat one not encoded")


