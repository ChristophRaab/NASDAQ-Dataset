import pandas as pd

if __name__ == '__main__':
    # Add your own full input and output path
    input_path = "Tweets.csv"
    output_path = "skipgram_metadata.tsv"

    csv_file = pd.read_csv(input_path)
    csv_file.to_csv(path_or_buf=output_path, sep='\t', header=None, index=None)
