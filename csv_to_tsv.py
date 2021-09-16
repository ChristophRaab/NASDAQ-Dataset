import csv
from pathlib import Path

# this reads the files that are in the same directory
csv_file = Path(__file__).with_name('Tweets.csv')
tsv_file = Path(__file__).with_name('Tweets.tsv')

# read every line from the comma separated file and save it in a tab separted file
with csv_file.open('r', encoding='utf8') as csvin, tsv_file.open('w', encoding='utf8') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in csvin:
        tsvout.writerow(row)