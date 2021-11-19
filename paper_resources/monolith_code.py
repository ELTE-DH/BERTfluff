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


def guess_kenlm(word: str, left_context: str, right_context: str, _: int) -> List[str]:
    missing_word = '#' * len(word)
    local_params = {'no_of_subwords': 1,
                    'prev_guesses[]': [],
                    'retry_wrong': False,
                    'top_n': 10,
                    'guesser': 'kenlm',
                    'missing_token': missing_word,
                    'contexts[]': [' '.join([left_context, missing_word, right_context])]}
    response = requests.post(f'{SERVER}/guess', json=local_params)
    return response.json()['guesses']


def guess_bert(word: str, left_context: str, right_context: str, no_subwords: int) -> List[str]:
    missing_word = '#' * len(word)
    payload = {'no_of_subwords': no_subwords,
               'prev_guesses[]': [],
               'retry_wrong': False,
               'top_n': 10,
               'guesser': 'bert',
               'missing_token': missing_word,
               'contexts[]': [' '.join([left_context, missing_word, right_context])]}
    response = requests.post(f'{SERVER}/guess', json=payload)
    return response.json()['guesses']


def make_context_length_measurement_both_side(word: str, left_context: List[str], right_context: List[str],
                                              no_subwords: int):
    context_max_length = len(left_context)
    context_min_length = 1
    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []

    for context_size in range(context_min_length, context_max_length):
        left, right = ' '.join(left_context[-context_size:]), ' '.join(right_context[:context_size])
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left, right, no_subwords)
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left, right, no_subwords)
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank}

    return output


def main():
    file_limit = 1000
    sample_size = 1000
    random.seed(42069)
    left_context = 6
    right_context = 6
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

    context_bank: List[Tuple[str, Tuple[str], Tuple[str], int]] = []
    with gzip.open('shuffled_final.txt.gz', 'rt') as infile:
        for i, line in tqdm.tqdm(enumerate(infile), total=file_limit):
            if i >= file_limit:
                break

            sentence = line.strip().split(' ')
            n_grams = get_ngram(sentence, left_cont_len=left_context, right_cont_len=right_context)
            for n_gram in n_grams:
                word, left, right = n_gram
                no_subwords = len(tokenizer(word, add_special_tokens=False)['input_ids'])
                context_bank.append((word, left, right, no_subwords))

    print(f'Number of contexts: {len(context_bank)}')
    print(f'Using first {sample_size}')

    contexts = context_bank[:sample_size]


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

    contextbank = []
    pool = multiprocessing.Pool(processes=128)
    results = list(tqdm.tqdm(pool.imap(guess_kenlm, contextbank), total=len(contextbank)))

    with open('kenlm_1long100_largemodel_100k_guesses.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for result, context in zip(results, contextbank):
            csv_writer.writerow(result + context)
    print(1)


if __name__ == '__main__':
    main()
