from typing import Tuple, Generator, List


def both_side(left_context: Tuple[str], right_context: Tuple[str]) -> Generator[Tuple[str, str], None, None]:
    """Increment both side word-by-word starting by one until the sorter side runs out of words"""
    left_context_rev = list(reversed(left_context))
    min_len = min(len(left_context), len(right_context))
    for i in range(1, min_len):
        left, right = left_context_rev[:i], right_context[:i]
        yield i, ' '.join(reversed(left)), ' '.join(right)


def complex_tactic(left_context: Tuple[str], right_context: Tuple[str], tactic: str) \
        -> Generator[Tuple[str, str], None, None]:
    left_size = 0
    right_size = 0
    for i, step in enumerate(tactic.split('|')):
        left_size += step.count('l')
        right_size += step.count('r')
        right = ' '.join(right_context[:right_size])
        left = ' '.join(left_context[-left_size:]) if left_size else ''
        yield i, left, right


def multi_guess_tactic(left_context: Tuple[str], right_context: Tuple[str], tactic: str) \
        -> Generator[Tuple[str, str], None, None]:
    """
    Takes care of the multi-context guesses. The inner state is kept with the `guess_w_guesser` function in the
    `guesser_comparator` module. Warning: this has a different signature than the rest of the tactics! The `left_context`
    and `right_context` variables contain entire left-right contexts. This is a hack.
    :param left_context: Tuple of left contexts.
    :param right_context: Tuple of right contexts.
    :param tactic: Unused.
    :return:
    """
    for i, (left, right) in enumerate(zip(left_context, right_context)):
        yield i, left, right
