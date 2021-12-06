from typing import Tuple, List
from itertools import repeat
from multiprocessing import Pool
from json import dump as json_dump
from argparse import ArgumentParser
from random import seed as random_seed


from tqdm import tqdm

from paper_resources.api import guess_kenlm, guess_bert
from paper_resources.context_bank_mimic import read_frequencies, sample_contexts


def make_context_length_measurement_both_side(args: Tuple[str, Tuple[str], Tuple[str], int, int, str, bool, bool]):
    word, left_context, right_context, no_subwords, index, tactic, store_previous, multi_guess = args

    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []
    for i in range(1, len(left_context)):
        left, right = ' '.join(left_context[-i:]), ' '.join(right_context[:i])

        bert_context_need, kenlm_context_need = guess_w_guessers(word, left, right, no_subwords, i, store_previous,
                                                                 left_prev_contexts, right_prev_contexts, multi_guess,
                                                                 bert_context_need, bert_guesses, bert_rank,
                                                                 kenlm_context_need, kenlm_guesses, kenlm_rank)

        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank, 'bert_output': bert_guesses,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank, 'kenlm_output': kenlm_guesses,
              'input': args}

    return output


def context_length_measurement_both_side_conc(args: Tuple[str, Tuple[str], Tuple[str], int, int, str, bool, bool]):
    word, left_context, right_context, no_subwords, index, tactic, store_previous, multi_guess = args

    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []
    for i, (left, right) in enumerate(args[['left_context', 'right_context']].values, start=1):

        bert_context_need, kenlm_context_need = guess_w_guessers(word, left, right, no_subwords, i, store_previous,
                                                                 left_prev_contexts, right_prev_contexts, multi_guess,
                                                                 bert_context_need, bert_guesses, bert_rank,
                                                                 kenlm_context_need, kenlm_guesses, kenlm_rank)

        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank, 'bert_output': bert_guesses,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank, 'kenlm_output': kenlm_guesses,
              'input': args}

    return output


def context_length_measurement_tactic_conc(args: Tuple[str, Tuple[str], Tuple[str], int, int, str, bool, bool]):
    word, left_context, right_context, no_subwords, index, tactic, store_previous, multi_guess = args

    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []
    for i, (left_full, right_full) in enumerate(args[['left_context', 'right_context']].values, start=1):
        left_size = tactic.count('l')
        right_size = tactic.count('r')
        right = ' '.join(left_full[:right_size])
        left = ' '.join(right_full[-left_size:]) if left_size else ''

        bert_context_need, kenlm_context_need = guess_w_guessers(word, left, right, no_subwords, i, store_previous,
                                                                 left_prev_contexts, right_prev_contexts, multi_guess,
                                                                 bert_context_need, bert_guesses, bert_rank,
                                                                 kenlm_context_need, kenlm_guesses, kenlm_rank)

        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank, 'bert_output': bert_guesses,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank, 'kenlm_output': kenlm_guesses,
              'input': args}

    return output


def tactic_1_one_left_one_right(args: Tuple[str, Tuple[str], Tuple[str], int, int, str, bool, bool]):
    word, left_context, right_context, no_subwords, index, tactic, store_previous, multi_guess = args

    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []

    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []
    for i, _ in enumerate(tactic, start=1):
        left_size = tactic[:i].count('l')
        right_size = tactic[:i].count('r')
        right = ' '.join(right_context[:right_size])
        left = ' '.join(left_context[-left_size:]) if left_size else ''

        bert_context_need, kenlm_context_need = guess_w_guessers(word, left, right, no_subwords, i, store_previous,
                                                                 left_prev_contexts, right_prev_contexts, multi_guess,
                                                                 bert_context_need, bert_guesses, bert_rank,
                                                                 kenlm_context_need, kenlm_guesses, kenlm_rank)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank, 'bert_output': bert_guesses,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank, 'kenlm_output': kenlm_guesses,
              'input': args}

    return output


def guess_w_guessers(word, left, right, no_subwords, i, store_previous, left_prev_contexts, right_prev_contexts,
                     multi_guess, bert_context_need, bert_guesses, bert_rank, kenlm_context_need, kenlm_guesses,
                     kenlm_rank):
    left_prev_contexts.append(left)
    right_prev_contexts.append(right)
    left_contexts = left_prev_contexts if multi_guess else [left]
    right_contexts = right_prev_contexts if multi_guess else [right]

    bert_context_need = guess_fun(word, left_contexts, right_contexts, no_subwords, i, store_previous,
                                  bert_context_need, bert_guesses, bert_rank, guess_bert)

    kenlm_context_need = guess_fun(word, left_contexts, right_contexts, no_subwords, i, store_previous,
                                   kenlm_context_need, kenlm_guesses, kenlm_rank, guess_kenlm)

    return bert_context_need, kenlm_context_need


def guess_fun(word, left_contexts, right_contexts, no_subwords, i, store_previous, context_need, guesses, rank,
              guesser):
    if context_need == -1:
        guess = guesser(word, left_contexts, right_contexts, no_subwords,
                        [guess[0] for guess in guesses] if store_previous else [])
        guesses.append(guess)
        if guess[0] == word:
            context_need = i
        rank.append(guess.index(word) if word in guess else -1)
    return context_need


def make_full_tactic(tactic: str, max_con: int) -> str:
    if tactic == 'both':
        return tactic
    else:
        rep = max(tactic.count('l'), tactic.count('r'))
        return (max_con // rep) * tactic


def main():
    args = parse_args()
    sample_size = args['sample_size']
    multi_guess = args['multi_guess']
    store_previous = args['store_previous']
    left_context = args['context_size']
    right_context = args['context_size']
    group_min = args['multi_concord']
    random_seed(42069)
    tactic = make_full_tactic(args['side'], left_context)
    print(f'Full tactic is {tactic}')

    read_frequencies('../resources/webcorp_2_freqs.tsv')

    contexts = [(*context, *boilerplate) for context, boilerplate in
                zip(sample_contexts(left_context, right_context, group_min, sample_size),
                    repeat((tactic, store_previous, multi_guess)))
                ]

    print(f'Number of contexts: {len(contexts)}')

    if tactic == 'both':
        if group_min > 0:
            func = context_length_measurement_both_side_conc
        else:
            func = make_context_length_measurement_both_side
    else:
        if group_min > 0:
            func = context_length_measurement_tactic_conc
        else:
            func = tactic_1_one_left_one_right

    results = exec_fun_for_contexts(func, contexts, args['n_jobs'])

    with open(f'{tactic}_context_{sample_size}_multi_{group_min}.json', 'w') as outfile:
        json_dump(results, outfile, ensure_ascii=False)
    print(1)


def exec_fun_for_contexts(func, contexts, n_jobs):
    if n_jobs == 1:
        results = [func(context) for context in contexts]
    else:
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(pool.imap_unordered(func, contexts), total=len(contexts)))
    return results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--context_size', type=int)
    parser.add_argument('--side', type=str)
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--n_jobs', type=int, default=64)
    parser.add_argument('--store_previous', action='store_true')
    parser.add_argument('--multi_guess', action='store_true')
    parser.add_argument('--multi_concord', type=int, default=0)
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    main()
