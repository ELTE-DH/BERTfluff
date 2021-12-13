from typing import Tuple, Generator


def both_side(left_context: Tuple[str], right_context: Tuple[str]) -> Generator[Tuple[str, str], None, None]:
    """Increment both side word-by-word starting by one until the sorter side runs out of words"""
    left_context_rev = list(reversed(left_context))
    min_len = min(len(left_context), len(right_context))
    for i in range(1, min_len):
        left, right = left_context_rev[:i], right_context[:i]
        yield i, ' '.join(reversed(left)), ' '.join(right)


def both_side_conc(left_context: Tuple[str], right_context: Tuple[str]) \
        -> Generator[Tuple[str, str], None, None]:
    """TODO ???"""
    for i, (left, right) in enumerate(args[['left_context', 'right_context']].values, start=1):
        yield i, left, right


def tactic_conc(left_context: Tuple[str], right_context: Tuple[str]) \
        -> Generator[Tuple[str, str], None, None]:
    """TODO ???"""
    for i, (left_full, right_full) in enumerate(args[['left_context', 'right_context']].values, start=1):
        left_size = tactic.count('l')
        right_size = tactic.count('r')
        right = ' '.join(left_full[:right_size])
        left = ' '.join(right_full[-left_size:]) if left_size else ''
        yield i, left, right


def one_left_one_right(left_context: Tuple[str], right_context: Tuple[str]) \
        -> Generator[Tuple[str, str], None, None]:
    """TODO ???"""
    # TODO itt ki van facsarva a jobb oldali kontextus. Szerintem nem jó.
    for i, _ in enumerate(tactic, start=1):
        left_size = tactic[:i].count('l')
        right_size = tactic[:i].count('r')
        right = ' '.join(right_context[:right_size])
        left = ' '.join(left_context[-left_size:]) if left_size else ''
        yield i, left, right
