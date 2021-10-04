import csv
from collections import Counter
from typing import Generator, Tuple, List
from random import choice as random_choice

from transformers import AutoTokenizer

# suppress warning, only complain when halting
# transformers.logging.set_verbosity_error()


# TODO Ezt nem használja semmi. -> Utilsba?
def create_corpora():
    """
    Used to create frequency. It also deduplicates in a rudimentary manner.
    """
    c = Counter()
    sentences = set()
    dupes = 0
    with open('resources/100k_tok.spl', encoding='UTF-8') as infile, \
            open('resources/tokenized_100k_corp.spl', 'w', encoding='UTF-8') as outfile:
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

    with open('resources/freqs.csv', 'w', encoding='UTF-8') as outfile:
        csv_writer = csv.writer(outfile)
        for line in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            csv_writer.writerow(line)  # (word, freq)


class ContextBank:
    def __init__(self, freqs_fn: str, corp_fn: str,):
        """

        :param freqs_fn: Word frequencies to choose.
        :param corp_fn: Corpus in SPL format.
        :return:
        """

        self.counter = self._create_counter(filename=freqs_fn)
        self.corp_fn = corp_fn
        self.tokenizer = AutoTokenizer.from_pretrained('models/hubert-base-cc', lowercase=True)

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

    def line_yielder(self, word: str, full_sentence: bool, window_size: int = 5) -> Generator[str, None, None]:
        """
        With a word_id, it starts tokenizing the corpora (which is fast, hence not precomputed),
        and when finds a sentence containing the token, it yields the sentence.
        :param window_size: Size of the window.
        :param word: word
        :param full_sentence: whether to return full sentence or only a context
        :return: A generator generating sentences or contexts
        """
        with open(self.corp_fn, encoding='UTF-8') as f:
            for line in f:
                sentence = line.strip().split(' ')
                if word in sentence:  # TODO ne a végén legyen, mert akkor nincs bal vagy jobb kontextus!
                    if full_sentence:
                        yield line.strip(), self._mask_sentence(line, word)
                    else:
                        # In order to create a not subword-based context, we have to first reconstruct
                        #  the original sentence, then find the word containing the subword,
                        #  then rebuild and return the context.
                        # Return a part of the original sentence containing the target word in the center
                        center = sentence.index(word)  # Returns first occurrence
                        context = sentence[max(0, center - window_size):min(len(sentence), center + window_size + 1)]
                        yield ' '.join(context), self._mask_sentence(' '.join(context), word)
                else:
                    continue

    def select_word(self, number_of_subwords: int) -> Tuple[str, list, int]:
        """
        Selects a word from the self.counter dictionary.

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
    def _mask_sentence(original_sentence: str, missing_word: str) -> Tuple[List[str], str]:
        """
        Masks a sentence in two ways: by replacing the selected word with `MISSING`, which is processed by the guessers
        and by replacing every character in the missing word with a hashmark.
        :param original_sentence: The tokenized sentence t omask.
        :param missing_word: The missing word.
        :return: The masked sentence to use with a `Guesser` and a pretty sentence for readability.
        """

        tokenized_sentence = original_sentence.split(' ')
        current_sentence = [word if word != missing_word else '#' * len(missing_word) for word in tokenized_sentence]

        mask_loc = tokenized_sentence.index(missing_word)
        masked_sentence = tokenized_sentence.copy()
        masked_sentence[mask_loc] = 'MISSING'

        return masked_sentence, ' '.join(current_sentence)
