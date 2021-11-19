import collections
import csv
import gzip
import multiprocessing
import random
from itertools import islice, tee
from string import ascii_lowercase
from typing import Tuple, List

import requests
import tqdm
from transformers import AutoTokenizer

LOWERCASE_LETTERS_HUN = set(ascii_lowercase + 'áéíóöőúüű')
NON_WORDS = set()
FREQUENCIES = collections.defaultdict(int)
FREQ_LOWER_LIMIT = 100
SERVER = 'http://127.0.0.1:8000'
SAMPLE_SIZE = 100

with open('non_words.txt', encoding='UTF-8') as fh:
    for elem in fh:
        elem = elem.rstrip()
        NON_WORDS.add(elem)


def n_gram_iter(input_iterator, n):
    return zip(*(islice(it, i, None) for i, it in enumerate(tee(iter(input_iterator), n))))


def get_ngram(sentence: List[List[str]], word_min_len=6, word_max_len=15,
              left_cont_len=5, right_cont_len=5) -> List[Tuple[str, Tuple[str], Tuple[str]]]:
    ngram_len = left_cont_len + 1 + right_cont_len
    right_cont_index = left_cont_len + 1
    possible_ngrams = []
    for entry in n_gram_iter(sentence, ngram_len):
        pre: Tuple[str] = entry[:left_cont_len]
        word = entry[left_cont_len]
        post: Tuple[str] = entry[right_cont_index:]
        if word_min_len <= len(word) <= word_max_len and LOWERCASE_LETTERS_HUN.issuperset(word) and \
                word not in NON_WORDS and FREQUENCIES[word] >= FREQ_LOWER_LIMIT:
            possible_ngrams.append((word, pre, post))

    return possible_ngrams


def guess_kenlm(args: List[str]) -> bool:
    word, left_context, right_context = args
    missing_word = '#' * len(word)
    local_params = {'no_of_subwords': 1,
                    'prev_guesses[]': [],
                    'retry_wrong': False,
                    'top_n': 5,
                    'guesser': 'kenlm',
                    'missing_token': missing_word,
                    'contexts[]': [' '.join([left_context, missing_word, right_context])]}
    response = requests.post(f'{SERVER}/guess', json=local_params)
    return response.json()['guesses']


def guess_bert(args: Tuple[str, Tuple[str], Tuple[str], int]):
    word, left_context, right_context, no_subwords = args
    missing_word = '#' * len(word)
    payload = {'no_of_subwords': no_subwords,
               'prev_guesses[]': [],
               'retry_wrong': False,
               'top_n': 5,
               'guesser': 'bert',
               'missing_token': missing_word,
               'contexts[]': [' '.join([left_context, missing_word, right_context])]}
    response = requests.post(f'{SERVER}/guess', json=payload)
    return response.json()['guesses']


def main():
    random.seed(42069)
    left_context = 5
    right_context = 5
    tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")

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

    contexts: List[Tuple[str, Tuple[str], Tuple[str]]] = []
    with gzip.open('shuffled_final.txt.gz', 'rt') as infile:
        for i, line in tqdm.tqdm(enumerate(infile), total=SAMPLE_SIZE):
            if i >= SAMPLE_SIZE:
                break

            sentence = line.strip().split(' ')
            n_grams = get_ngram(sentence, left_cont_len=left_context, right_cont_len=right_context)
            for n_gram in n_grams:
                word, left, right = n_gram
                no_subwords = tokenizer(word)

    with open('output.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for line in sorted(contexts):
            csv_writer.writerow(line)


def guess():
    params = {
        # 'guesser': guesser_type,
        # 'contexts[]': [' '.join(context) for context in context_1],
        'no_of_subwords': 1,
        'prev_guesses[]': [],
        'retry_wrong': False,
        'top_n': 5
    }

    data: List[Tuple[str, str, str, str]] = []

    random.seed(42069)

    with open('webcorp_1_contextbank_100.csv') as infile:
        csv_reader = csv.reader(infile)
        contextbank = random.sample([line for line in csv_reader], k=SAMPLE_SIZE)

    pool = multiprocessing.Pool(processes=128)
    results = list(tqdm.tqdm(pool.imap(guess_kenlm, contextbank), total=len(contextbank)))

    with open('kenlm_1long100_largemodel_100k_guesses.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for result, context in zip(results, contextbank):
            csv_writer.writerow(result + context)
    print(1)


if __name__ == '__main__':
    main()
