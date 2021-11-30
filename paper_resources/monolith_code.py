from typing import Tuple, List
from collections import Counter
from multiprocessing import Pool
from json import dump as json_dump
from argparse import ArgumentParser
from random import seed as random_seed

from tqdm import tqdm
import pandas as pd

from paper_resources.api import guess_kenlm, guess_bert
from paper_resources.context_bank_mimic import make_context_bank, read_frequencies


def make_context_length_measurement_both_side(args: Tuple[str, Tuple[str], Tuple[str], int, int, str, bool, bool]):
    word, left_context, right_context, no_subwords, index, tactic, store_previous, multi_guess = args
    context_max_length = len(left_context)
    context_min_length = 1
    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []
    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []

    for context_size in range(context_min_length, context_max_length):
        left, right = ' '.join(left_context[-context_size:]), ' '.join(right_context[:context_size])
        left_prev_contexts.append(left)
        right_prev_contexts.append(right)
        left_contexts = left_prev_contexts if multi_guess else [left]
        right_contexts = right_prev_contexts if multi_guess else [right]
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left_contexts, right_contexts, no_subwords,
                                    [guess[0] for guess in bert_guesses] if store_previous else [])
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = context_size
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left_contexts, right_contexts, no_subwords,
                                      [guess[0] for guess in kenlm_guesses] if store_previous else [])
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


def context_length_measurement_both_side_conc(args: pd.DataFrame):
    word, _, _, no_subwords, index, tactic, store_previous, multi_guess = args.iloc[0]

    # how long of a context each model needs
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []
    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []

    for i, (left, right) in enumerate(args[['left_context', 'right_context']].values, 1):
        left_prev_contexts.append(left)
        right_prev_contexts.append(right)
        left_contexts = left_prev_contexts if multi_guess else [left]
        right_contexts = right_prev_contexts if multi_guess else [right]
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left_contexts, right_contexts, no_subwords,
                                    [guess[0] for guess in bert_guesses] if store_previous else [])
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = i
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left_contexts, right_contexts, no_subwords,
                                      [guess[0] for guess in kenlm_guesses] if store_previous else [])
            kenlm_guesses.append(kenlm_guess)
            if kenlm_guess[0] == word:
                kenlm_context_need = i
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank,
              'input': args.to_dict(), 'bert_output': bert_guesses, 'kenlm_output': kenlm_guesses}

    return output


def context_length_measurement_tactic_conc(args: pd.DataFrame):
    word, _, _, no_subwords, index, tactic, store_previous, multi_guess = args.iloc[0]
    bert_context_need = -1
    kenlm_context_need = -1
    bert_rank: List[int] = []
    kenlm_rank: List[int] = []
    bert_guesses: List[List[str]] = []
    kenlm_guesses: List[List[str]] = []
    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []

    for i, (left_full, right_full) in enumerate(args[['left_context', 'right_context']].values, 1):
        left_size = tactic.count('l')
        right_size = tactic.count('r')
        right = ' '.join(left_full[:right_size])
        left = ' '.join(right_full[-left_size:]) if left_size else ''
        left_prev_contexts.append(left)
        right_prev_contexts.append(right)
        left_contexts = left_prev_contexts if multi_guess else [left]
        right_contexts = right_prev_contexts if multi_guess else [right]
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left_contexts, right_contexts, no_subwords,
                                    [guess[0] for guess in bert_guesses] if store_previous else [])
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = i
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left_contexts, right_contexts, no_subwords,
                                      [guess[0] for guess in kenlm_guesses] if store_previous else [])
            kenlm_guesses.append(kenlm_guess)
            if kenlm_guess[0] == word:
                kenlm_context_need = i
            kenlm_rank.append(kenlm_guess.index(word) if word in kenlm_guess else -1)
        # if both have guessed
        if kenlm_context_need != -1 and bert_context_need != -1:
            break

    output = {'bert_guess': bert_context_need, 'bert_rank': bert_rank,
              'kenlm_guess': kenlm_context_need, 'kenlm_rank': kenlm_rank,
              'input': args.to_dict(), 'bert_output': bert_guesses, 'kenlm_output': kenlm_guesses}

    return output


def tactic_1_one_left_one_right(args: Tuple[str, Tuple[str], Tuple[str], int, int, str, bool, bool]):
    word, left_context, right_context, no_subwords, index, tactic, store_previous, multi_guess = args
    context_max_length = len(left_context)
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
        left_prev_contexts.append(left)
        right_prev_contexts.append(right)
        left_contexts = left_prev_contexts if multi_guess else [left]
        right_contexts = right_prev_contexts if multi_guess else [right]
        if bert_context_need == -1:
            bert_guess = guess_bert(word, left_contexts, right_contexts, no_subwords,
                                    [guess[0] for guess in bert_guesses] if store_previous else [])
            bert_guesses.append(bert_guess)
            if bert_guess[0] == word:
                bert_context_need = i
            bert_rank.append(bert_guess.index(word) if word in bert_guess else -1)
        if kenlm_context_need == -1:
            kenlm_guess = guess_kenlm(word, left_contexts, right_contexts, no_subwords,
                                      [guess[0] for guess in kenlm_guesses] if store_previous else [])
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
    random_seed(42069)
    left_context = args['context_size']
    right_context = args['context_size']
    group_min = args['multi_concord']
    tactic = make_full_tactic(args['side'], left_context)
    print(f'Full tactic is {tactic}')

    read_frequencies('../resources/webcorp_2_freqs.tsv')

    contexts = []
    if group_min:
        raw_contexts = []
        with tqdm(total=sample_size) as pbar:
            for i, context in enumerate(make_context_bank(left_context, right_context)):
                raw_contexts.append(context + (i, tactic, store_previous, multi_guess))
                if i % 1000 == 0 and i != 0:
                    c = Counter(i[0] for i in raw_contexts)
                    no_concordances = sum(freq > group_min for freq in c.values())
                    pbar.update(no_concordances - pbar.n)
                    if no_concordances >= sample_size:
                        break

        df = pd.DataFrame(raw_contexts,
                          columns=['word', 'left_context', 'right_context', 'subwords', 'index', 'tactic',
                                   'store_previous', 'multi_guess'])
        for word, group in df.groupby('word'):
            if len(group) >= group_min:
                contexts.append(group.iloc[:10])
            if len(contexts) >= sample_size:
                break

    else:
        for i, context in tqdm(enumerate(make_context_bank(left_context, right_context)), total=sample_size):
            contexts.append(context + (i, tactic, store_previous, multi_guess))
            if len(contexts) >= sample_size:
                break

    print(f'Number of contexts: {len(contexts)}')

    if tactic == 'both':
        if group_min:
            func = context_length_measurement_both_side_conc
        else:
            func = make_context_length_measurement_both_side
    else:
        if group_min:
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
