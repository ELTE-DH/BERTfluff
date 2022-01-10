from itertools import repeat
from multiprocessing import Pool
from typing import Tuple, Callable, Generator, List, Dict, Any

import requests
from tqdm import tqdm


def exec_fun_for_contexts(contexts: List[Tuple[str, Tuple[str], Tuple[str], int, int]],
                          boilerplate: Tuple[Callable[[Tuple[str], Tuple[str], str], Generator[Tuple[str, str], None, None]],
                                             bool, bool, str, Tuple[str, str], str], n_jobs: int):
    """Prepares data for multiprocessing and hands to a remote server.

    :param contexts:  A list of tuples, where each tuple is a context (middle, left, right words, number of subwords,
     context index
    :param boilerplate:  Tuple: tactics function, whether to include previous guesses, whether to use multiple
    contexts for each guess, server address, and the name of the guessers to use
    :param n_jobs: Number of parallel jobs. Should match the number of gunicorn workers on the server side.
    :return: A dictionary containing all data that can be used to recreate the experiment.
    """
    contexts_w_boilerplate = ((word, left, right, no_subwords, i,
                               tactic_fun, store_previous, multi_guess, server_addr, guesser_names, tactics)
                              for (word, left, right, no_subwords, i),
                                  (tactic_fun, store_previous, multi_guess, server_addr, guesser_names, tactics)
                              in zip(contexts, repeat(boilerplate)))
    if n_jobs == 1:
        results = [context_length_measurement(context) for context in contexts_w_boilerplate]
    else:
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(pool.imap_unordered(context_length_measurement, contexts_w_boilerplate),
                                total=len(contexts)))
    return results


def context_length_measurement(args: Tuple[str, Tuple[str], Tuple[str], int, int,
                                           Callable[[Tuple[str], Tuple[str], str], Generator[Tuple[str, str], None, None]],
                                           bool, bool, str, Tuple[str, str], str]):
    """Runs one experiment on a concordance.

    :param args: Tuple: word, left_context, right_context, no_subwords, line index, tactics function, store_previous,
    whether to use multiple contexts for guessing, server address
    :return:
    """
    word, left_context, right_context, no_subwords, index, tactic_fun, store_previous, multi_guess, server_addr, \
    guesser_names, tactics = args

    guessers = {}
    # this dictionary gets loaded up with results in the `guess_w_guessers` function
    for guesser_name in guesser_names:
        guessers[guesser_name] = {
            'guess': -1,  # How long context each model needs
            'rank': [],  # List[int]
            'output': []  # List[List[str]]
        }

    left_prev_contexts: List[str] = []
    right_prev_contexts: List[str] = []
    for i, left, right in tactic_fun(left_context, right_context, tactics):
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


def guess_w_guessers(word: str, left: str, right: str, no_subwords: int, i: int, store_previous: bool,
                     multi_guess: bool, server_addr: str, left_prev_contexts: List[str], right_prev_contexts: List[str],
                     guessers: Dict[str, Any]) -> List[int]:
    """Runs a guess through the REST API. Modifies `guessers` in place: loads it up with the guesses.

    :param word: The missing word.
    :param left: The current left context.
    :param right: The current right context.
    :param no_subwords: Number of subwords (needed for BERT)
    :param i: The index of the context.
    :param store_previous: Whether to retry the earlier guesses or not.
    :param left_prev_contexts: Previous left contexts.
    :param right_prev_contexts: Previous right contexts.
    :param multi_guess: Whether to take into account multiple contexts or just the latest.
    :param server_addr: Address to send the request to.
    :param guessers: Guessers to ask from the server.
    :return: A list of ranks - out of all the guesses, the correct word was in which position (first, second, fifth...).
    """
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
            guesses = guess_rest_api(server_addr, guesser_name, word, left_contexts, right_contexts, no_subwords,
                                     prev_guesses)
            if guesses[0] == word:
                curr_context_need = i
            else:
                curr_context_need = -1

            context_need.append(curr_context_need)
            # Update params for output
            params['guess'] = curr_context_need
            params['output'].append(guesses)
            params['rank'].append(guesses.index(word) if word in guesses else -1)

    return context_need


def guess_rest_api(server_addr: str, guesser_name: str, word: str, left_context: List[str], right_context: List[str],
                   no_subwords: int = 1, previous_guesses: List[str] = None) -> List[str]:
    """Simple wrapper for the REST API. Sends the data in the correct format.

    :param server_addr: Server address.
    :param guesser_name: Name of guesser. Usually `KenLM` or `BERT`.
    :param word: Missing word. Used for calculating the length.
    :param left_context: Left contexts.
    :param right_context: Right contexts.
    :param no_subwords: Number of subwords, only used in BERT.
    :param previous_guesses: List containing the previous guesses.
    :return:
    """
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
