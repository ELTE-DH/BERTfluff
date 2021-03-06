from typing import Tuple, Generator


def complex_tactic(left_context: Tuple[str], right_context: Tuple[str], tactic: str) \
        -> Generator[Tuple[str, str], None, None]:
    # TODO rewrite to match the input of multi_guess_tactic (tuple of contexts != tuple of words)
    # TODO comment, examples!
    left_size = 0
    right_size = 0
    for i, step in enumerate(tactic.split('|')):
        left_size += step.count('l')
        right_size += step.count('r')
        right = ' '.join(right_context[:right_size])
        left = ' '.join(left_context[-left_size:]) if left_size else ''
        yield i, left, right


def multi_guess_tactic(left_context: Tuple[str], right_context: Tuple[str], _: str) \
        -> Generator[Tuple[str, str], None, None]:
    """
    Takes care of the multi-context guesses. The inner state is kept with the `guess_w_guesser` function in the
    `guesser_comparator` module. Warning: this has a different signature than the rest of the tactics! The `left_context`
    and `right_context` variables contain entire left-right contexts. This is a hack.
    :param left_context: Tuple of left contexts.
    :param right_context: Tuple of right contexts.
    :param _: (tactic) Unused.
    :return:
    """
    # TODO rewrite this to match the the input of the two other tactics (tuple of contexts != tuple of words)
    for i, (left, right) in enumerate(zip(left_context, right_context)):
        yield i, left, right
