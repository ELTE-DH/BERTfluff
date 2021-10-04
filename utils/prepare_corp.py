import csv
import os
from collections import Counter


def create_counters(corp_fp: str, wordlist_fp: str):
    """

    Creates helper files based on an spl file.

    :param corp_fp: Path to corpus.
    :param wordlist_fp: Path to wordlist.
    :return:
    """

    counter = Counter()
    with open(os.path.expanduser(corp_fp)) as infile:
        for line in infile:
            for word in line.strip().split(' '):
                counter[word] += 1

    with open(os.path.expanduser(wordlist_fp), 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for word, freq in sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True):
            csv_writer.writerow([word, freq])


def filter_wordlist(in_fp: str, out_fp: str):
    """
    Filters a wordlist. A wordlist is only accessed by the guesser to build the tree, thus can be provided from a
    much larger corpus than the guessing game's corpus.

    :param in_fp: Input filepath.
    :param out_fp: Output filepath.
    :return: None
    """

    words = set()

    with open(in_fp) as infile:
        for line in infile:
            word = line.strip()
            if word.isalpha() and word.islower():
                words.add(word)

    with open(out_fp, 'w') as outfile:
        for word in sorted(words):
            outfile.write(f'{word}\n')


if __name__ == '__main__':
    create_counters('../resources/tokenized_100k_corp.spl', '../resources/freqs.csv')
    filter_wordlist('../resources/wordlist_3M_unfiltered.csv', '../resources/wordlist_3M.csv')
