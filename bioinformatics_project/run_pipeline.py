from data_retrieval_manipulation import *

if __name__ == '__main__':
    promoters_data, enhancers_data = download_epigenomic_data()
    print('Data obtained: promoters and enhancers epigenomes and labels')

    hg19 = download_genome_data()
    print('Data obtained: genome hg19 - ' + str(hg19))

    promoters_sequence_data = extract_sequence_data(hg19, promoters_data["data"])
    print("Data obtained: promoters sequence data")

    enhancers_sequence_data = extract_sequence_data(hg19, enhancers_data["data"])
    print("Data obtained: enhancers sequence data")
