from json import dump as json_dump

from guesser_comparator import guess_rest_api
import sqlite3
from transformers import BertTokenizer
from multiprocessing import Pool

from tqdm import tqdm


def guess_wrapper(args):

    server_addr, guesser_name, word, left_context, right_context, no_subwords, previous_guesses, id_ = args
    results = guess_rest_api(server_addr, guesser_name, word, left_context, right_context, no_subwords, previous_guesses)
    return id_, guesser_name, word, left_context, right_context, no_subwords, results


def main():
    tokenizer = BertTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
    con = sqlite3.connect('/home/levaid/Projects/word-guessing-game/webcorpus1_conts.db')
    cur = con.cursor()
    parallel_data = []
    for row in cur.execute('select id, left, word, right, freq, sent from lines'):
        id_, left, word, right, freq, sent = row
        no_subwords = len(tokenizer(word, add_special_tokens=False)['input_ids'])
        for guesser in ['bert', 'kenlm']:
            parallel_data.append(('http://127.0.0.1:42069', guesser, word, [left], [right], no_subwords, [], id_))

    n_jobs = 96
    # guess_rest_api(*parallel_data[0])
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(guess_wrapper, parallel_data),
                            total=len(parallel_data)))

    with open('human_vs_machine.json', 'w') as outfile:
        json_dump(results, outfile, ensure_ascii=False)


if __name__ == '__main__':
    main()
