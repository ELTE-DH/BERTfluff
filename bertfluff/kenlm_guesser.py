from collections import defaultdict
from typing import List, Tuple

import kenlm


class KenLMGuesser:
    def __init__(self, model_path: str = 'models/10M_pruned.bin',
                 vocabulary_path: str = 'resources/wordlist_3M.csv'):
        self.model = kenlm.Model(model_path)
        self.vocabulary = defaultdict(list)
        with open(vocabulary_path) as infile:
            for line in map(str.strip, infile):
                self.vocabulary[len(line)].append(line)

    def _predict_logprob(self, target: str, left_context: str, right_context: str):
        return self.model.score(f'{left_context} {target} {right_context}', bos=True, eos=True)

    def make_guess(self, contexts: List[Tuple[str, str, str]], number_of_subwords: int,
                   previous_guesses: List[str], retry_wrong: bool = False, top_n: int = 10) -> List[str]:
        """
        A gensim-based guesser. Since gensim's API is stable, it can be either FastText, CBOW or skip-gram, as long
        as the model has a `predict_output_word` method.
        Takes the top 1_000_000 guesses for each context, creates the intersection of these guesses, and returns
        the word with the highest possibility by taking the product of the possibilities for each context.

        :param contexts: contexts
        :param number_of_subwords: number of subwords to guess (not used here)
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :return: cbow guesses in a list
        """
        probs = defaultdict(float)
        for context in contexts:
            for word in self.vocabulary[len(context[1])]:
                probs[word] += self._predict_logprob(word, context[0], context[2])

        guesses = []

        for word, prob in sorted(probs.items(), key=lambda x: (-x[1], x[0])):
            if retry_wrong or word not in previous_guesses:
                guesses.append((word, prob))
            if len(guesses) >= top_n:
                break

        return [guess[0] for guess in guesses]

    @staticmethod
    def split_to_subwords(selected_word: str) -> List[int]:
        """
        Split the word to subwords returning word_ids (dummy function)

        :param selected_word:
        :return: the selected word, the word_ids and the frequency of the selected word
        """

        _ = selected_word
        return [0]
