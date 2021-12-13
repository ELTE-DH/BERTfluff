import csv
import gzip
import random

import tqdm

from paper_resources.context_bank_mimic import read_frequencies_and_nonwords, get_ngram


def wirte_contexts_csv(fname, contexts):
    with open(fname, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for pre, word, post in sorted(contexts):
            csv_writer.writerow((word, ' '.join(pre), ' '.join(post)))


def main():
    random.seed(42069)
    read_frequencies_and_nonwords('../resources/webcorp_2_freqs.tsv', 'non_words.txt')

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
