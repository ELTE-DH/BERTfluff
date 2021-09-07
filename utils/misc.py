import urllib.parse
from typing import List, Iterable


def bert_get_convert(contexts: List[List[str]], word_length: int, number_of_subwords: int,
                     previous_guesses: Iterable[str], retry_wrong: bool, top_n: int,
                     missing_token: str) -> str:
    str_contexts = urllib.parse.quote('#'.join('_'.join(context) for context in contexts))
    str_prev_guesses = urllib.parse.quote('_'.join(previous_guesses)) if previous_guesses != [] else '_'
    str_retry_wrong = 'y' if retry_wrong else 'n'

    extension = f'{str_contexts}/{word_length}/{number_of_subwords}/' + \
                f'{str_prev_guesses}/{str_retry_wrong}/{top_n}/{missing_token}'

    return extension
