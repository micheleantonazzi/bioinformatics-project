from epigenomic_dataset import load_epigenomes


def download_data(cell_line='HEK293', window_size=200):
    promoters_data, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )

    enhancers_data, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )


if __name__ == "__main__":
    download_data()
