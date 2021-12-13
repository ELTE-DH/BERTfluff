from itertools import repeat
from multiprocessing import Pool
from typing import Tuple, Callable, Generator, List

import requests
from tqdm import tqdm


def exec_fun_for_contexts(contexts: List[Tuple[str, Tuple[str], Tuple[str], int, int]],
                          boilerplate: Tuple[Callable[[Tuple[str], Tuple[str]], Generator[Tuple[str, str], None, None]],
                                             bool, bool, str, Tuple[str, ...]], n_jobs: int):
    contexts_w_boilerplate = ((word, left, right, no_subwords, i,
                               tactic_fun, store_previous, multi_guess, server_addr, guesser_names)
                              for (word, left, right, no_subwords, i),
                                  (tactic_fun, store_previous, multi_guess, server_addr, guesser_names)
                              in zip(contexts, repeat(boilerplate)))
    if n_jobs == 1:
        results = [context_length_measurement(context) for context in contexts_w_boilerplate]
    else:
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(pool.imap_unordered(context_length_measurement, contexts_w_boilerplate),
                                total=len(contexts)))
    return results


def context_length_measurement(args: Tuple[str, Tuple[str], Tuple[str], int, int,
                                           Callable[[Tuple[str], Tuple[str]], Generator[Tuple[str, str], None, None]],
                                           bool, bool, str, Tuple[str, ...]]):
    word, left_context, right_context, no_subwords, index, tactic_fun, store_previous, multi_guess, server_addr, \
        guesser_names = args

    guessers = {}
    for guesser_name in guesser_names:
        guessers[guesser_name] = {
            'guess': -1,  # How long context each model needs
            'rank': [],  # List[int]
            'output': []  # List[List[str]]
        }

    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []
    for i, left, right in tactic_fun(left_context, right_context):
        context_need = guess_w_guessers(word, left, right, no_subwords, i, store_previous, multi_guess, server_addr,
                                        left_prev_contexts, right_prev_contexts, guessers)

        # If all have guessed
        if all(cn != -1 for cn in context_need):
            break

    output = {'gussers': guessers,
              'args': {'word': word, 'left_context': left_context, 'right_context': right_context,
                       'no_subwords': no_subwords, 'index': index, 'tactic': tactic_fun.__name__,
                       'store_previous': store_previous, 'multi_guess': multi_guess, 'server_addr': server_addr}}

    return output


def guess_w_guessers(word, left, right, no_subwords, i, store_previous, left_prev_contexts, right_prev_contexts,
                     multi_guess, server_addr, guessers):
    if multi_guess:
        left_prev_contexts.append(left)
        right_prev_contexts.append(right)
        left_contexts = left_prev_contexts
        right_contexts = right_prev_contexts
    else:
        left_contexts, right_contexts = [left], [right]

    context_need = []
    for guesser_name, params in guessers.items():
        if params['guess'] == -1:  # Guesser have guessed already?
            if store_previous:
                prev_guesses = [guess[0] for guess in params['output']]
            else:
                prev_guesses = []
            guess = guess_rest_api(server_addr, guesser_name, word, left_contexts, right_contexts, no_subwords,
                                   prev_guesses)
            if guess[0] == word:
                curr_context_need = i
            else:
                curr_context_need = -1

            context_need.append(curr_context_need)
            # Update params for output
            params['guess'] = curr_context_need
            params['output'].append(guess)
            params['rank'].append(guess.index(word) if word in guess else -1)

    return context_need


def guess_rest_api(server_addr: str, guesser_name: str, word: str, left_context: List[str], right_context: List[str],
                   no_subwords: int = 1, previous_guesses: List[str] = None) -> List[str]:
    if previous_guesses is None:
        previous_guesses = []
    missing_word = '#' * len(word)
    local_params = {'no_of_subwords': no_subwords,
                    'prev_guesses[]': previous_guesses,
                    'retry_wrong': False,
                    'top_n': 10,
                    'guesser': guesser_name,
                    'missing_token': missing_word,
                    'contexts[]': [f'{left} {missing_word} {right}'
                                   for left, right in zip(left_context, right_context)]}
    response = requests.post(f'{server_addr}/guess', json=local_params)
    return response.json()['guesses']
