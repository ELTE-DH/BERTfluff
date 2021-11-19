import csv
import gzip
import random
import sys
from itertools import islice, tee
from string import ascii_lowercase
from typing import Tuple, List
import collections

import tqdm

LOWERCASE_LETTERS_HUN = set(ascii_lowercase + 'áéíóöőúüű')
NON_WORDS = set()
FREQUENCIES = collections.defaultdict(int)
FREQ_LOWER_LIMIT = 100

with open('non_words.txt', encoding='UTF-8') as fh:
    for elem in fh:
        elem = elem.rstrip()
        NON_WORDS.add(elem)


def n_gram_iter(input_iterator, n):
    return zip(*(islice(it, i, None) for i, it in enumerate(tee(iter(input_iterator), n))))


def get_ngram(sentence, word_min_len=6, word_max_len=15,
              left_cont_len=5, right_cont_len=5):
    ngram_len = left_cont_len + 1 + right_cont_len
    right_cont_index = left_cont_len + 1
    possible_ngrams = []
    for entry in n_gram_iter(sentence, ngram_len):
        pre = entry[:left_cont_len]
        word = entry[left_cont_len]
        post = entry[right_cont_index:]
        if word_min_len <= len(word) <= word_max_len and LOWERCASE_LETTERS_HUN.issuperset(word) and \
                word not in NON_WORDS and FREQUENCIES[word] >= FREQ_LOWER_LIMIT:
            possible_ngrams.append((word, ' '.join(pre), ' '.join(post)))

    return possible_ngrams


def main():
    random.seed(42069)
    with open('../resources/webcorp_2_freqs.tsv') as infile:
        for line in infile:
            try:
                word, freq = line.strip().split('\t')
                freq = int(freq)
                FREQUENCIES[word.strip()] = freq
                if freq < FREQ_LOWER_LIMIT:
                    break
            except ValueError:
                continue

    contexts: List[Tuple[str, str, str]] = []
    with gzip.open('filter_final.txt.gz', 'rt') as infile:
        for line in tqdm.tqdm(infile, total=13915132):
            sentence = line.strip().split(' ')
            left_context = random.randint(0, 2)
            right_context = 2 - left_context
            if left_context == 0 and right_context == 0:
                continue
            possible_ngrams = get_ngram(sentence, left_cont_len=left_context, right_cont_len=right_context)
            if possible_ngrams:
                ngram = random.choice(possible_ngrams)
                contexts.append(ngram)

    with open('webcorp_1_contextbank_100.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for line in sorted(contexts):
            csv_writer.writerow(line)


if __name__ == '__main__':
    main()
