import csv
from collections import Counter
from os.path import join as os_path_join
from typing import Generator, Tuple
from random import choice as random_choice


class ContextBank:
    def __init__(self, freqs_fn: str, corp_fn: str, resources_dir: str = 'resources'):
        """
        Interface for selecting words and appropriate contexts for them  # TODO SQL backend?

        :param freqs_fn: Word frequencies to choose.
        :param corp_fn: Corpus in SPL format.
        :param resources_dir: The directory where the resources are stored.
        :return:
        """

        self.counter = self._create_counter(os_path_join(resources_dir, freqs_fn))
        self.corp_fn = os_path_join(resources_dir, corp_fn)

    @staticmethod
    def _create_counter(filename: str, min_threshold: int = 30) -> Counter:
        """
        Create a word frequency counter from file

        :param filename:
        :param min_threshold:
        :return:
        """
        c = Counter()
        with open(filename, encoding='UTF-8') as infile:
            csv_reader = csv.reader(infile)
            for word, freq in csv_reader:
                if int(freq) < min_threshold or len(word) <= 5:
                    continue
                else:
                    c[word] = int(freq)
        return c

    def get_examples(self, word: str, window_size: int = 5, hide_char: str = '#') -> Generator[(str, str, str)]:
        """
        Yield an example context from the corpus which contains the selected word
         (full sentence or window sized context)

        :param word: word
        :param window_size: Size of the window negative numbers returns the full sentence
        :param: hide_char: The character used to hide the word
        :return: A generator generating sentences or contexts
        """

        if window_size >= 2:
            context_size = (window_size - 1) // 2
        else:
            context_size = 1_000_000  # Extremely big to include full sentence

        with open(self.corp_fn, encoding='UTF-8') as f:
            for i, line in enumerate(f):  # TODO the order of contexts is deterministic!
                sentence = line.strip().split(' ')
                # Find word in sentence...
                word_index = next((i for i, x in enumerate(sentence) if x == word), -1)
                # ... and have at least window_size context to the left and right as well!
                if word_index > 0 and context_size <= word_index <= len(sentence) - 1 - context_size:
                    left_truncated = ' '.join(sentence[max(word_index-context_size, 0):word_index])
                    right_truncated = ' '.join(sentence[word_index+1:word_index+1+context_size])

                    yield left_truncated, hide_char * len(word), right_truncated

    def select_word(self) -> Tuple[str, int]:
        """
        Selects a random word from the self.counter dictionary

        :return: the selected word, the word_ids and the frequency of the selected word
        """

        selected_word = random_choice(list(self.counter.keys()))
        selected_word_freq = self.counter[selected_word]

        return selected_word, selected_word_freq
