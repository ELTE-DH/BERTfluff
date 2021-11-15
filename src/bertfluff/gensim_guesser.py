import logging
from collections import defaultdict
from itertools import chain, islice
from os.path import join as os_path_join
from typing import List, Tuple

import gensim


class GensimGuesser:
    def __init__(self, model_fn='hu_fasttext_100.gensim', models_dir='models'):
        # self.model = gensim.models.Word2Vec.load(os_path_join(models_dir, model_fn))
        self.model = gensim.models.Word2Vec.load(os_path_join(models_dir, model_fn), mmap='r')

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

        _ = number_of_subwords
        word_length = len(contexts[0][1])
        fixed_contexts = [list(chain(left, right)) for left, _, right in contexts]

        # We build the initial wordlist from the first guess
        # TODO 1_000_000 volt ígérve! Magyarázat?
        init_probabilities = {word: prob for word, prob in self.model.predict_output_word(fixed_contexts[0], 10_000)
                              if len(word) == word_length and (word not in previous_guesses or retry_wrong)}

        # And we filter it in the following guesses.
        probabilities = defaultdict(lambda x: float(1.0), init_probabilities)
        for context in fixed_contexts[1:]:
            output_dict = dict(self.model.predict_output_word(context, 10_000))
            for word in probabilities:
                if word in output_dict:
                    probabilities[word] *= output_dict[word]
                else:
                    probabilities[word] = 0

        # Sort results by descending probabilities
        words = list(islice((w for w, _ in sorted(probabilities.items(), key=lambda x: (-x[1], x[0]))), top_n))

        return words

    @staticmethod
    def split_to_subwords(selected_word: str) -> List[int]:
        """
        Split the word to subwords returning word_ids (dummy function)

        :param selected_word:
        :return: the selected word, the word_ids and the frequency of the selected word
        """

        _ = selected_word
        return [0]

    @staticmethod
    def prepare_resources(input_corpus: str, resources_dir: str = 'resources', models_dir: str = 'models'):
        logging.basicConfig(filename='gensim.log',
                            format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.DEBUG)
        sentences = gensim.models.word2vec.LineSentence(os_path_join(resources_dir, input_corpus))
        model = gensim.models.Word2Vec(max_vocab_size=1_000_000, workers=8, window=5, vector_size=100, sg=0)
        model.build_vocab(sentences)
        model.train(sentences, epochs=1, total_examples=model.corpus_count)
        model.save(os_path_join(models_dir, 'hu_wv.gensim'))

        return model

    def word_similarity(self, word_1: str, word_2: str) -> float:
        """
        Calculates similarity by taking the cosine similarity of the vectors of `word_1` and `word_2`.
        Due to the use of FastText, every word has a vector representation.

        :param word_1: One word
        :param word_2: The other word
        :return: Similarity of the corresponding word vectors.
        """

        return self.model.wv.similarity(word_1, word_2)


def download(input_corpus='corp_10M.spl'):
    GensimGuesser.prepare_resources(input_corpus)


if __name__ == '__main__':
    # just download and build everything if the module is not imported
    download()
