import argparse
import collections
import gzip
import json
import multiprocessing
import os
import random
from itertools import islice, tee
from string import ascii_lowercase
from typing import Tuple, List, Iterable, Iterator

import requests
import tqdm
from transformers import AutoTokenizer

LOWERCASE_LETTERS_HUN = set(ascii_lowercase + 'áéíóöőúüű')
NON_WORDS = set()
FREQUENCIES = collections.defaultdict(int)
FREQ_LOWER_LIMIT = 100
SERVER = 'http://127.0.0.1:8000'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def make_context_bank(left_max_context: int, right_max_context: int) \
        -> Iterator[Tuple[str, Tuple[str], Tuple[str], int]]:
    tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
    with gzip.open('shuffled_final.txt.gz', 'rt') as infile:
        for line in infile:
            sentence = line.strip().split(' ')
            n_grams = get_ngram(sentence, left_cont_len=left_max_context, right_cont_len=right_max_context)
            for n_gram in n_grams:
                word, left, right = n_gram
                no_subwords = len(tokenizer(word, add_special_tokens=False)['input_ids'])
                yield word, left, right, no_subwords


def make_context_length_measurement_both_side(args: Tuple[str, Tuple[str], Tuple[str], int, int]):
    word, left_context, right_context, no_subwords = args
    context_max_length = len(left_context)
    context_min_length = 1
    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    for context_size in range(context_min_length, context_max_length):
        left, right = ' '.join(left_context[-context_size:]), ' '.join(right_context[:context_size])
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left, right, no_subwords)
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = context_size
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left, right, no_subwords)
            kenlm_guesses.append(kenlm_guess)
            if kenlm_guess[0] == word:
                kenlm_context_need = context_size
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank,
              'input': args, 'bert_output': bert_guesses, 'kenlm_output': kenlm_guesses}

    return output


def make_right_context_measurement(args: Tuple[str, Tuple[str], Tuple[str], int, int]):
    word, left_context, right_context, no_subwords = args
    context_max_length = len(right_context)
    context_min_length = 1
    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    for context_size in range(context_min_length, context_max_length):
        left = ''
        right = ' '.join(right_context[:context_size])
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left, right, no_subwords)
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = context_size
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left, right, no_subwords)
            kenlm_guesses.append(kenlm_guess)
            if kenlm_guess[0] == word:
                kenlm_context_need = context_size
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank,
              'input': args, 'bert_output': bert_guesses, 'kenlm_output': kenlm_guesses}

    return output


def make_left_context_measurement(args: Tuple[str, Tuple[str], Tuple[str], int, int]):
    word, left_context, right_context, no_subwords, index = args
    context_max_length = len(left_context)
    context_min_length = 1
    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    for context_size in range(context_min_length, context_max_length):
        left = ' '.join(left_context[-context_size:])
        right = ''
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left, right, no_subwords)
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = context_size
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left, right, no_subwords)
            kenlm_guesses.append(kenlm_guess)
            if kenlm_guess[0] == word:
                kenlm_context_need = context_size
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank,
              'input': args, 'bert_output': bert_guesses, 'kenlm_output': kenlm_guesses}

    return output


def tactic_1_one_left_one_right(args: Tuple[str, Tuple[str], Tuple[str], int, int, str]):
    word, left_context, right_context, no_subwords, index, tactic = args
    context_max_length = len(left_context)
    full_tactic = make_full_tactic(tactic, context_max_length)
    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    for i, _ in enumerate(full_tactic, 1):
        left_size = full_tactic[:i].count('l')
        right_size = full_tactic[:i].count('r')
        right = ' '.join(right_context[:right_size])
        left = ' '.join(left_context[-left_size:]) if left_size else ''
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left, right, no_subwords)
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = i
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left, right, no_subwords)
            kenlm_guesses.append(kenlm_guess)
            if kenlm_guess[0] == word:
                kenlm_context_need = i
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break


    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank,
              'input': args, 'bert_output': bert_guesses, 'kenlm_output': kenlm_guesses}

    return output


def make_full_tactic(tactic: str, max_con: int) -> str:
    rep = max([tactic.count('l'), tactic.count('r')])
    return (max_con // rep) * tactic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_size', type=int)
    parser.add_argument('--side', type=str)
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--n_jobs', type=int, default=64)
    args = vars(parser.parse_args())
    side = args['side']
    sample_size = args['sample_size']
    random.seed(42069)
    left_context = args['context_size']
    right_context = args['context_size']

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

    contexts = []
    for i, context in tqdm.tqdm(enumerate(make_context_bank(left_context, right_context)), total=sample_size):
        contexts.append(context + (i,))
        if len(contexts) >= sample_size:
            break
    print(f'Number of contexts: {len(contexts)}')

    func_map = {'left': make_left_context_measurement,
                'both': make_context_length_measurement_both_side,
                'right': make_right_context_measurement}

    if side in ['left', 'both', 'right']:
        func = func_map['side']
    else:
        contexts = [context + (args['side'],) for context in contexts]
        func = tactic_1_one_left_one_right

    pool = multiprocessing.Pool(processes=args['n_jobs'])
    results = list(
        tqdm.tqdm(pool.imap_unordered(func, contexts), total=len(contexts)))

    with open(f'{side}_context_{sample_size}.json', 'w') as outfile:
        json.dump(results, outfile, ensure_ascii=False)
    print(1)


if __name__ == '__main__':
    main()
