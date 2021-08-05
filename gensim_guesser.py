from typing import List, Iterable
from collections import defaultdict

import gensim


class GensimGuesser:
    def __init__(self, model_fn='models/hu_wv.gensim'):
        self.model = gensim.models.Word2Vec.load(model_fn)

    @staticmethod
    def create_compatible_context(context: List[str], missing_token: str) -> List[str]:

        return [word for word in context if word != missing_token]

    def make_guess(self, contexts: List[List[str]], word_length: int, number_of_subwords: int,
                   previous_guesses: Iterable[str], retry_wrong: bool, top_n: int = 10,
                   missing_token: str = 'MISSING') -> List[str]:

        fixed_contexts = [self.create_compatible_context(context, missing_token) for context in contexts]
        guess_vocab = set()
        probabilities = defaultdict(lambda: 1.0)

        for context in fixed_contexts:
            current_vocab = set()
            for word, prob in self.model.predict_output_word(context, 1_000_000):
                if len(word) == word_length:
                    current_vocab.add(word)
                    probabilities[word] *= prob

            if len(guess_vocab) == 0:
                guess_vocab = current_vocab.copy()
            else:
                guess_vocab = guess_vocab.intersection(current_vocab)

        guesses = {word: probabilities[word] for word in guess_vocab}

        retval = []
        for word, prob in sorted(guesses.items(), key=lambda x: x[1], reverse=True):
            if word not in previous_guesses or retry_wrong:
                retval.append(word)
            if len(retval) >= top_n:
                break

        return retval
