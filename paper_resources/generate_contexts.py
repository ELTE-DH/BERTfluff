import gzip
import random

import tqdm

from paper_resources.context_bank_mimic import read_frequencies, get_ngram, wirte_contexts_csv


def main():
    random.seed(42069)
    read_frequencies('../resources/webcorp_2_freqs.tsv')

    contexts = []
    with gzip.open('filter_final.txt.gz', 'rt') as infile:
        for line in tqdm.tqdm(infile, total=13915132):
            sentence = line.strip().split(' ')

            left_context = random.randint(0, 2)
            right_context = 2 - left_context

            possible_ngrams = get_ngram(sentence, left_cont_len=left_context, right_cont_len=right_context)

            if len(possible_ngrams) > 0:
                ngram = random.choice(possible_ngrams)
                contexts.append(ngram)

    wirte_contexts_csv('webcorp_1_contextbank_100.csv', contexts)


if __name__ == '__main__':
    main()
