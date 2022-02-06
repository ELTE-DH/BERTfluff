from argparse import ArgumentParser
from json import dump as json_dump
from random import seed as random_seed

from paper_resources.context_bank_mimic import sample_contexts
from paper_resources.guesser_comparator import exec_fun_for_contexts
from paper_resources.tactics import complex_tactic, multi_guess_tactic


def main():
    args = parse_args()
    server_addr: str = args['server_addr']
    sample_size = args['sample_size']
    multi_guess: bool = args['multi_guess']
    store_previous: bool = args['store_previous']
    left_context_size: int = args['left_context_size']
    right_context_size: int = args['right_context_size']
    group_min: int = args['multi_concord']
    n_jobs: int = args['n_jobs']
    tactic: str = args['tactic']
    stored_rand_seed: int = args['random_seed']
    freq_filename: str = args['freq_filename']
    non_words_filename: str = args['non_words']

    random_seed(stored_rand_seed)  # restore random seed

    # TODO ??? Comment!
    con_size = max(left_context_size, right_context_size)

    rep = max(tactic.count('l'), tactic.count('r'))
    full_tactic = '|'.join((con_size // rep) * [tactic])

    print(f'Full tactic is {full_tactic}')

    contexts = list(sample_contexts(freq_filename, non_words_filename, left_context_size, right_context_size, group_min,
                                    sample_size))

    # print(f'Number of contexts: {len(contexts)}')

    if args['multi_guess']:
        tactic_func = multi_guess_tactic
    else:
        tactic_func = complex_tactic

    boilerplate_for_contexts = (tactic_func, store_previous, multi_guess, server_addr, ('bert', 'kenlm'),
                                full_tactic)
    results = exec_fun_for_contexts(contexts, boilerplate_for_contexts, n_jobs)

    with open(f'{tactic}_context_{sample_size}_multi_{group_min}.json', 'w') as outfile:
        json_dump(results, outfile, ensure_ascii=False)
    print('Finished!')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--left-context_size', type=int, required=True)
    parser.add_argument('--right-context_size', type=int, required=True)
    parser.add_argument('--tactic', type=str, required=True)
    parser.add_argument('--sample_size', type=int, required=True)
    parser.add_argument('--n_jobs', type=int, default=64)
    parser.add_argument('--store_previous', action='store_true')
    parser.add_argument('--multi_guess', action='store_true')
    parser.add_argument('--multi_concord', type=int, default=0)
    parser.add_argument('--server-addr', type=str, default='http://127.0.0.1:8000')
    parser.add_argument('--random-seed', type=int, default=42069)
    parser.add_argument('--freq-filename', type=str, default='../resources/webcorp_2_freqs.tsv', required=True)
    parser.add_argument('--non-words', type=str, default='non_words.txt', required=True)
    parser.add_argument('--guesser', nargs='+', help='Name of the guessers separated by spaces (bert, kenlm))',
                        default=['bert', 'kenlm'], required=True)
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    main()
