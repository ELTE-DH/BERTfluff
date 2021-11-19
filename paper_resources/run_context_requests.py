import csv
import multiprocessing
import random
from typing import List, Tuple

import requests
import tqdm

SERVER = 'http://127.0.0.1:8000'


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


def main():
    params = {
        # 'guesser': guesser_type,
        # 'contexts[]': [' '.join(context) for context in context_1],
        'no_of_subwords': 1,
        'prev_guesses[]': [],
        'retry_wrong': False,
        'top_n': 5
    }
    sample_size = 100_000
    data: List[Tuple[str, str, str, str]] = []

    random.seed(42069)

    with open('webcorp_1_contextbank_100.csv') as infile:
        csv_reader = csv.reader(infile)
        contextbank = random.sample([line for line in csv_reader], k=sample_size)

    pool = multiprocessing.Pool(processes=128)
    results = list(tqdm.tqdm(pool.imap(guess_kenlm, contextbank), total=len(contextbank)))

    with open('kenlm_1long100_largemodel_100k_guesses.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for result, context in zip(results, contextbank):
            csv_writer.writerow(result + context)
    print(1)

if __name__ == '__main__':
    main()
