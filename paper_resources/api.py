from typing import List

import requests

SERVER = 'http://127.0.0.1:8000'


def guess_kenlm(word: str, left_context: List[str], right_context: List[str], _: int,
                previous_guesses: List[str] = None) \
        -> List[str]:
    if previous_guesses is None:
        previous_guesses = []
    missing_word = '#' * len(word)
    local_params = {'no_of_subwords': 1,
                    'prev_guesses[]': previous_guesses,
                    'retry_wrong': False,
                    'top_n': 10,
                    'guesser': 'kenlm',
                    'missing_token': missing_word,
                    'contexts[]': [f'{left} {missing_word} {right}' for left, right in
                                   zip(left_context, right_context)]}
    response = requests.post(f'{SERVER}/guess', json=local_params)
    return response.json()['guesses']


def guess_bert(word: str, left_context: List[str], right_context: List[str], no_subwords: int,
               previous_guesses: List[str] = None) -> List[str]:
    if previous_guesses is None:
        previous_guesses = []
    missing_word = '#' * len(word)
    payload = {'no_of_subwords': int(no_subwords),
               'prev_guesses[]': previous_guesses,
               'retry_wrong': False,
               'top_n': 10,
               'guesser': 'bert',
               'missing_token': missing_word,
               'contexts[]': [f'{left} {missing_word} {right}' for left, right in zip(left_context, right_context)]}
    response = requests.post(f'{SERVER}/guess', json=payload)
    return response.json()['guesses']