import csv
from collections import defaultdict


def create_counters(corp_fp: str, wordlist_fp: str):
    """

    Creates helper files based on an spl file.

    :param corp_fp:
    :param wordlist_fp:
    :return:
    """

    counter = defaultdict(int)
    with open(corp_fp) as infile:
        for line in infile:
            for word in line.strip().split(' '):
                counter[word.lower()] += 1

    with open(wordlist_fp, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for word, freq in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            if word.isalpha():
                csv_writer.writerow([word, freq])


def filter_wordlist(in_fp: str, out_fp: str):
    """
    Filters a wordlist. A wordlist is only accessed by the guesser to build the tree.

    :param in_fp: Input filepath.
    :param out_fp: Output filepath.
    :return: None
    """

    words = set()

    with open(in_fp) as infile:
        for word in map(str.strip, infile):
            if word.isalpha():
                words.add(word.lower())

    with open(out_fp, 'w') as outfile:
        for word in sorted(words):
            outfile.write(f'{word}\n')


if __name__ == '__main__':
    create_counters('../resources/tokenized_100k_corp.spl', '../resources/freqs.csv')
    filter_wordlist('../resources/wordlist_3M_unfiltered.csv', '../resources/wordlist_3M.csv')
