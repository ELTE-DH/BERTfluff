import os
import gzip
import collections
from itertools import islice, tee
from string import ascii_lowercase
from typing import List, Tuple, Iterator

from transformers import AutoTokenizer

LOWERCASE_LETTERS_HUN = set(ascii_lowercase + 'áéíóöőúüű')
FREQUENCIES = collections.defaultdict(int)
FREQ_LOWER_LIMIT = 100

with open('non_words.txt', encoding='UTF-8') as fh:
    NON_WORDS = set()
    for elem in fh:
        elem = elem.rstrip()
        NON_WORDS.add(elem)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _n_gram_iter(input_iterator, n):
    return zip(*(islice(it, i, None) for i, it in enumerate(tee(iter(input_iterator), n))))


def _get_ngram(sentence: List[List[str]], word_min_len=6, word_max_len=15,
               left_cont_len=5, right_cont_len=5) -> List[Tuple[str, Tuple[str], Tuple[str]]]:
    ngram_len = left_cont_len + 1 + right_cont_len
    right_cont_index = left_cont_len + 1
    possible_ngrams = []
    for entry in _n_gram_iter(sentence, ngram_len):
        pre: Tuple[str] = entry[:left_cont_len]
        word = entry[left_cont_len]
        post: Tuple[str] = entry[right_cont_index:]
        if word_min_len <= len(word) <= word_max_len and LOWERCASE_LETTERS_HUN.issuperset(word) and \
                word not in NON_WORDS and FREQUENCIES[word] >= FREQ_LOWER_LIMIT:
            possible_ngrams.append((word, pre, post))

    return possible_ngrams


def make_context_bank(left_max_context: int, right_max_context: int) \
        -> Iterator[Tuple[str, Tuple[str], Tuple[str], int]]:
    tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
    with gzip.open('shuffled_final.txt.gz', 'rt') as infile:
        for line in infile:
            sentence = line.strip().split(' ')
            n_grams = _get_ngram(sentence, left_cont_len=left_max_context, right_cont_len=right_max_context)
            for n_gram in n_grams:
                word, left, right = n_gram
                no_subwords = len(tokenizer(word, add_special_tokens=False)['input_ids'])
                yield word, left, right, no_subwords


def read_frequencies(infile_name):
    with open(infile_name) as infile:
        for line in infile:
            try:
                word, freq = line.strip().split('\t')
                freq = int(freq)
                FREQUENCIES[word.strip()] = freq
                if freq < FREQ_LOWER_LIMIT:
                    break
            except ValueError:
                continue
