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

    with open(os_path_expanduser(corp_fp)) as infile, open(os_path_expanduser(wordlist_fp), 'w') as outfile:
        counter = Counter(word for line in infile for word in line.strip().split(' '))
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

    with open(in_fp, encoding='UTF-8') as infile, open(out_fp, 'w', encoding='UTF-8') as outfile:
        words = {word for word in map(str.strip, infile) if word.isalpha() and word.islower()}
        for word in sorted(words):
            print(word, file=outfile)


def create_corpora(resources_dir: str = 'resources'):
    """
    Used to create frequency. It also deduplicates in a rudimentary manner.
    """

    with open(os_path_join(resources_dir, '100k_tok.spl'), encoding='UTF-8') as infile, \
            open(os_path_join(resources_dir, 'tokenized_100k_corp.spl'), 'w', encoding='UTF-8') as sents_outfile, \
            open(os_path_join(resources_dir, 'freqs.csv'), 'w', encoding='UTF-8') as freqs_outfile:
        c = Counter()
        sentences = set()
        dupes = 0
        for line in infile:
            if line[0:2] == '# ':
                continue
            sentence = tuple(line.strip().split(' '))
            if sentence not in sentences:
                sentences.add(sentence)
                for token in sentence:
                    c[token] += 1
                print(line, end='', file=sents_outfile)  # line is not stripped, hence we print it as-is
            else:
                dupes += 1

        print(f'There were {dupes} duplicated sentences.')

        csv_writer = csv.writer(freqs_outfile)
        for line in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            csv_writer.writerow(line)  # (word, freq)


if __name__ == '__main__':
    create_counters('resources/tokenized_100k_corp.spl', 'resources/freqs.csv')
    filter_wordlist('resources/wordlist_3M_unfiltered.csv', 'resources/wordlist_3M.csv')
