import csv
from collections import Counter
from os.path import join as os_path_join
from typing import Generator, Tuple, List
from random import choice as random_choice

from transformers import AutoTokenizer

# suppress warning, only complain when halting
# transformers.logging.set_verbosity_error()


# TODO Ezt nem használja semmi. -> Utilsba?
def create_corpora(resources_dir: str = 'resources'):
    """
    Used to create frequency. It also deduplicates in a rudimentary manner.
    """
    c = Counter()
    sentences = set()
    dupes = 0
    with open(os_path_join(resources_dir, '100k_tok.spl'), encoding='UTF-8') as infile, \
            open(os_path_join(resources_dir, 'tokenized_100k_corp.spl'), 'w', encoding='UTF-8') as outfile:
        for line in infile:
            if line[0] == '#':
                continue
            sentence = tuple(line.strip().split(' '))
            if sentence not in sentences:
                sentences.add(sentence)
            else:
                dupes += 1
                continue

            for token in sentence:
                c[token] += 1
            print(line, end='', file=outfile)

    print(f'There were {dupes} duplicated sentences.')

    with open(os_path_join(resources_dir, 'freqs.csv'), 'w', encoding='UTF-8') as outfile:
        csv_writer = csv.writer(outfile)
        for line in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            csv_writer.writerow(line)  # (word, freq)


class ContextBank:
    def __init__(self, freqs_fn: str, corp_fn: str, resources_dir: str = 'resources', models_dir: str = 'models'):
        """

        :param freqs_fn: Word frequencies to choose.
        :param corp_fn: Corpus in SPL format.
        :param resources_dir: The directory where the resources are stored.
        :param models_dir: The directory where the models are stored.
        :return:
        """

        self.counter = self._create_counter(os_path_join(resources_dir, freqs_fn))
        self.corp_fn = os_path_join(resources_dir, corp_fn)
        self.tokenizer = AutoTokenizer.from_pretrained(os_path_join(models_dir, 'hubert-base-cc'), lowercase=True)

    @staticmethod
    def _create_counter(filename: str, min_threshold: int = 30) -> Counter:
        """

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

    @staticmethod
    def _create_context(word: str, sentence: List[str], window_size: int = 5, line: str = ''):
        """Return a part of the original sentence containing the target word in the center"""
        center = sentence.index(word)  # Returns first occurrence
        context = sentence[max(0, center - window_size):min(len(sentence), center + window_size + 1)]
        return ' '.join(context)

    @staticmethod
    def _full_sentence_as_context(word: str, sentence: List[str], window_size: int = 5, line: str = ''):
        return line

    # TODO Itt a docstirng hülyeségeket ír!
    def line_yielder(self, word: str, full_sentence: bool, window_size: int = 5) -> Generator[str, None, None]:
        """
        In order to create a not subword-based context, we have to first reconstruct the original sentence,
         then find the word containing the subword, then rebuild and return the context.
        :param window_size: Size of the window.
        :param word: word
        :param full_sentence: whether to return full sentence or only a context
        :return: A generator generating sentences or contexts
        """
        if full_sentence:
            context_creator_fun = self._full_sentence_as_context
        else:
            context_creator_fun = self._create_context

        with open(self.corp_fn, encoding='UTF-8') as f:
            for line in f:
                line = line.strip()
                sentence = line.split(' ')
                # Have at least window_size word context to the left and right as well!
                if word in sentence[window_size:-window_size]:
                    context_str = context_creator_fun(word, sentence, window_size, line)
                    yield context_str, self._mask_sentence(context_str, word)

    # TODO Itt a docstirng hülyeségeket ír!
    def select_word(self, number_of_subwords: int) -> Tuple[str, list, int]:
        """
        Selects a word from the self.counter dictionary.
        With a word_id, it starts tokenizing the corpora (which is fast, hence not precomputed),
         and when finds a sentence containing the token.
        :param number_of_subwords:
        :return:
        """

        selected_word, selected_input_ids = '', []
        while len(selected_input_ids) != number_of_subwords:
            selected_word = random_choice(list(self.counter.keys()))
            selected_input_ids = self.tokenizer(selected_word, add_special_tokens=False)['input_ids']

        selected_word_freq = self.counter[selected_word]

        return selected_word, selected_input_ids, selected_word_freq

    @staticmethod
    def _mask_sentence(original_sentence: str, missing_word: str) -> Tuple[str, List[str]]:
        """
        Masks a sentence in two ways: by replacing the selected word with `MISSING`, which is processed by the guessers
        and by replacing every character in the missing word with a hashmark.
        :param original_sentence: The tokenized sentence to mask.
        :param missing_word: The missing word.
        :return: The masked pretty sentence for readability and and a format to be used with a `Guesser`
        """

        tokenized_sentence = original_sentence.split(' ')
        current_sentence = [word if word != missing_word else '#' * len(missing_word) for word in tokenized_sentence]

        mask_loc = tokenized_sentence.index(missing_word)
        masked_sentence = tokenized_sentence.copy()
        masked_sentence[mask_loc] = 'MISSING'

        return ' '.join(current_sentence), masked_sentence
