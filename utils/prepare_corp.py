import csv
from collections import Counter
from os.path import join as os_path_join, expanduser as os_path_expanduser


def create_counters(corp_fp: str, wordlist_fp: str):
    """

    Creates helper files based on an spl file.

    :param corp_fp: Path to corpus.
    :param wordlist_fp: Path to wordlist.
    :return:
    """

    counter = Counter()
    with open(os_path_expanduser(corp_fp)) as infile:
        for line in infile:
            for word in line.strip().split(' '):
                counter[word] += 1

    with open(os_path_expanduser(wordlist_fp), 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for word_freq_pair in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            csv_writer.writerow(word_freq_pair)


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


def create_corpora(resources_dir: str = 'resources'):
    """
    Used to create frequency. It also deduplicates in a rudimentary manner.
    """
    c = Counter()
    sentences = set()
    dupes = 0
    with open(os_path_join(resources_dir, '100k_tok.spl'), encoding='UTF-8') as infile, \
            open(os_path_join(resources_dir, 'tokenized_100k_corp.spl'), 'w', encoding='UTF-8') as outfile:
        for line in infile:
            if line[0] == '#':
                continue
            sentence = tuple(line.strip().split(' '))
            if sentence not in sentences:
                sentences.add(sentence)
            else:
                dupes += 1
                continue

            for token in sentence:
                c[token] += 1
            print(line, end='', file=outfile)

    print(f'There were {dupes} duplicated sentences.')

    with open(os_path_join(resources_dir, 'freqs.csv'), 'w', encoding='UTF-8') as outfile:
        csv_writer = csv.writer(outfile)
        for line in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            csv_writer.writerow(line)  # (word, freq)


if __name__ == '__main__':
    create_counters('../resources/tokenized_100k_corp.spl', '../resources/freqs.csv')
    filter_wordlist('../resources/wordlist_3M_unfiltered.csv', '../resources/wordlist_3M.csv')
