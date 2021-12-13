
from json import dump as json_dump
from argparse import ArgumentParser
from random import seed as random_seed

from paper_resources.guesser_comparator import exec_fun_for_contexts
from paper_resources.context_bank_mimic import sample_contexts
from paper_resources.tactics import both_side, both_side_conc, tactic_conc, one_left_one_right


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
    side: str = args['side']
    stored_rand_seed: int = args['random_seed']
    freq_filename: str = args['freq_filename']
    non_words_filename: str = args['non_words']

    random_seed(stored_rand_seed)  # restore random seed

    # TODO ??? Comment!
    con_size = max(left_context_size, right_context_size)
    if side == 'both':
        tactic = side
    else:
        rep = max(side.count('l'), side.count('r'))
        tactic = (con_size // rep) * side

    print(f'Full tactic is {tactic}')

    # TODO Comment!
    if tactic == 'both':
        if group_min > 0:
            tactic_fun = both_side_conc
        else:
            tactic_fun = both_side
    else:
        if group_min > 0:
            tactic_fun = tactic_conc
        else:
            tactic_fun = one_left_one_right

    contexts = sample_contexts(freq_filename, non_words_filename, left_context_size, right_context_size, group_min,
                               sample_size)

    print(f'Number of contexts: {len(contexts)}')

    boilerplate_for_contexts = (tactic_fun, store_previous, multi_guess, server_addr, ('bert', 'kenlm'))
    results = exec_fun_for_contexts(contexts, boilerplate_for_contexts, n_jobs)

    with open(f'{tactic}_context_{sample_size}_multi_{group_min}.json', 'w') as outfile:
        json_dump(results, outfile, ensure_ascii=False)
    print('Finished!')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--left-context_size', type=int, required=True)
    parser.add_argument('--right-context_size', type=int, required=True)
    parser.add_argument('--side', type=str, required=True)
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
