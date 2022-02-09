import os
import gzip
import collections
from collections import defaultdict, Counter
from itertools import islice, tee
from string import ascii_lowercase
from typing import List, Tuple, Iterator

from tqdm import tqdm
from transformers import AutoTokenizer

LOWERCASE_LETTERS_HUN = set(ascii_lowercase + 'áéíóöőúüű')
FREQUENCIES = collections.defaultdict(int)
FREQ_LOWER_LIMIT = 100
NON_WORDS = set()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _n_gram_iter(input_iterator, n):
    return zip(*(islice(it, i, None) for i, it in enumerate(tee(iter(input_iterator), n))))


def get_ngram(sentence: List[List[str]], word_min_len=6, word_max_len=15,
              left_cont_len=5, right_cont_len=5) -> List[Tuple[str, Tuple[str], Tuple[str]]]:
    if left_cont_len == 0 and right_cont_len == 0:
        return []

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


def make_context_bank(left_max_context: int, right_max_context: int, fname='shuffled_final.txt.gz',
                      tokenizer_model='SZTAKI-HLT/hubert-base-cc') -> Iterator[Tuple[str, Tuple[str], Tuple[str], int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    with gzip.open(fname, 'rt') as infile:
        for line in infile:
            sentence: List[List[str]] = line.strip().split(' ')
            n_grams = get_ngram(sentence, left_cont_len=left_max_context, right_cont_len=right_max_context)
            for n_gram in n_grams:
                word, left, right = n_gram
                no_subwords = len(tokenizer(word, add_special_tokens=False)['input_ids'])
                yield word, left, right, no_subwords


def read_frequencies_and_nonwords(freq_filename, non_words_filename):
    with open(freq_filename) as freq_file:
        for line in freq_file:
            try:
                word, freq = line.strip().split('\t')
                freq = int(freq)
                FREQUENCIES[word.strip()] = freq
                if freq < FREQ_LOWER_LIMIT:
                    break
            except ValueError:
                continue
    with open(non_words_filename, encoding='UTF-8') as fh:
        for elem in fh:
            elem = elem.rstrip()
            NON_WORDS.add(elem)


def sample_contexts(freq_filename, non_words_filename, left_context_size, right_context_size, group_min_size,
                    sample_size):
    if len(FREQUENCIES) == 0:
        read_frequencies_and_nonwords(freq_filename, non_words_filename)

    # TODO this is broken - it should only yield 1 long list of group_min long lists, not group_min long seperate ones
    if group_min_size > 0:  # TODO Shouldn't this be > 1?
        words = set()
        conc_by_word = defaultdict(list)
        with tqdm(total=sample_size) as pbar:
            c = Counter()
            for i, (word, left, right, no_subwords) \
                    in enumerate(make_context_bank(left_context_size, right_context_size)):
                c[word] += 1
                conc_by_word[word].append((word, left, right, no_subwords, i))

                if i % 1000 == 0 and i != 0:
                    no_concordances = 0
                    words = set()
                    for no_concordances, (curr_word, size_of_concordance) in enumerate(c.most_common()):
                        if size_of_concordance < group_min_size or no_concordances >= sample_size:
                            break
                        words.add(curr_word)
                    else:
                        no_concordances += 1  # All good, have to add one because indexing started at 0!

                    pbar.update(no_concordances - pbar.n)
                    if no_concordances >= sample_size:
                        break

        for n, curr_word in enumerate(words):
            # HACK here
            # We need a tuple of strings where one string is one entire context, not just a word
            # This way, we _hack_ it into the API - the multi_concord tactic accounts for this and returns the
            # contexts unchanged.
            # current group is a 5-tuple
            current_group = conc_by_word[curr_word][:group_min_size]
            word, _, _, no_subwords, i = current_group[0]
            left_contexts = tuple(' '.join(context[1]) for context in current_group)
            right_contexts = tuple(' '.join(context[2]) for context in current_group)

            yield word, left_contexts, right_contexts, no_subwords, i

            if n >= sample_size:
                break

    else:
        for i, (word, left, right, no_subwords) \
                in tqdm(enumerate(make_context_bank(left_context_size, right_context_size), start=1),
                        total=sample_size):
            # TODO Shouldn't this return tuples also for uniform return types of the two branches in the if clause?
            #  e.g. yield word, (left,), (right,), no_subwords, i
            yield word, left, right, no_subwords, i
            if i >= sample_size:
                break
