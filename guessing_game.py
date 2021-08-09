import csv
import random
from collections import Counter
from typing import Generator, List
import transformers
import guessers


# suppress warning, only complain when halting
# transformers.logging.set_verbosity_error()


def create_corpora():
    """
    Used to create frequency. It also deduplicates in a rudimentary manner.
    """
    c = Counter()
    sentences = set()
    dupes = 0
    with open('resources/100k_tok.spl') as infile, open('resources/tokenized_100k_corp.spl', 'w') as outfile:
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

    with open('resources/freqs.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for word, freq in sorted(c.items(), key=lambda x: x[1], reverse=True):
            csv_writer.writerow([word, freq])


def create_aligned_text(sentences: List[str]) -> List[str]:
    hashmark_positions = [sen.find('#') for sen in sentences]
    zero_point = max(hashmark_positions)
    return [' ' * (zero_point - position) + sentence for position, sentence in zip(hashmark_positions, sentences)]


class Game:
    def __init__(self, freqs_fn: str, corp_fn: str, guesser):
        """

        :param freqs_fn: Word frequencies to choose.
        :param corp_fn: Corpus in SPL format.
        :return:
        """
        self.counter = self.create_counter(filename=freqs_fn)
        self.corp_fn = corp_fn
        self.guesser = guesser
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)

    @staticmethod
    def create_counter(filename: str, min_threshold: int = 30) -> Counter:
        """

        :param filename:
        :param min_threshold:
        :return:
        """
        c = Counter()
        with open(filename) as infile:
            csv_reader = csv.reader(infile)
            for word, freq in csv_reader:
                if int(freq) < min_threshold or not word.isalpha() or len(word) <= 5:
                    continue
                else:
                    c[word] = int(freq)
        return c

    def line_yielder(self, fname: str, word: str, full_sentence: bool,
                     window_size: int = 5) -> Generator[str, None, None]:
        """
        With a word_id, it starts tokenizing the corpora (which is fast, hence not precomputed),
        and when finds a sentence containing the token, it yields the sentence.
        :param window_size: Size of the window.
        :param fname: corpus file in spl format
        :param word:
        :param full_sentence:
        :return: A generator generating sentences or contexts
        """
        with open(fname) as f:
            for line in f:
                sentence = line.strip().split(' ')
                if word in sentence:
                    if full_sentence:
                        yield line.strip()
                    else:
                        yield self.create_context(sentence, word, window_size)
                else:
                    continue

    @staticmethod
    def create_context(sentence: List[str], target_word: str, window_size: int = 5) -> str:
        """
        In order to create a not subword-based context, we have to first reconstruct the original sentence,
        then find the word containing the subword, then rebuild and return the context.
        :param sentence: list of tokens
        :param target_word: target word
        :param window_size: size of the window
        :return: a part of the original sentence containing the target word in the center
        """

        center = sentence.index(target_word)  # returns first occurrence

        return ' '.join(sentence[max(0, center - window_size):min(len(sentence), center + window_size + 1)])

    def guessing_game(self, show_bert_output: bool = True, full_sentence: bool = False,
                      number_of_subwords: int = 1) -> List:
        """
        Provides the interface for the game.
        :return: a list of length 3, containing the number of guesses of the player, BERT and the word missing
        """
        while True:
            selected_word = random.choice(list(self.counter.keys()))
            selected_wordids = self.tokenizer(selected_word, add_special_tokens=False)
            if len(selected_wordids['input_ids']) == number_of_subwords:
                break

        guesses = set()
        user_guessed = False
        bert_guessed = False
        retval = [-1, -1, selected_word]
        sentences = []
        contexts = []

        # print(selected_word)
        print(len(selected_word), selected_wordids, self.counter[selected_word])

        for i, orig_sentence in enumerate(self.line_yielder(self.corp_fn, selected_word, full_sentence)):
            tokenized_sentence = orig_sentence.split(' ')
            mask_loc = tokenized_sentence.index(selected_word)

            masked_sentence = tokenized_sentence.copy()
            masked_sentence[mask_loc] = 'MISSING'

            contexts.append(masked_sentence)
            bert_guesses = self.guesser.make_guess(contexts, word_length=len(selected_word),
                                                   previous_guesses=guesses, retry_wrong=False,
                                                   number_of_subwords=len(selected_wordids['input_ids']))

            # UI
            current_sentence = orig_sentence.replace(selected_word, '#' * len(selected_word), 1)
            sentences.append(current_sentence)
            print('\n'.join(create_aligned_text(sentences)))
            print('-' * 80)

            if not user_guessed:
                user_input = input('Please input your guess: ')
                if user_input.strip() == selected_word:
                    user_guessed = True
                    retval[0] = i + 1
                elif user_input.strip() == '':
                    user_guessed = True

            print(f'Computer\'s guess is {bert_guesses[:1]}')

            if show_bert_output:
                print(f'Computer\'s top 10 guesses: {" ".join(bert_guesses[:10])}')

            guess = bert_guesses[0] if len(bert_guesses) > 0 else ''

            if not bert_guessed:
                if guess == selected_word:
                    print('Computer guessed the word.')
                    bert_guessed = True
                    retval[1] = i + 1
                else:
                    guesses.add(guess)

            if bert_guessed and user_guessed:
                return retval

        # in case player does not guess it, we return
        return retval


if __name__ == '__main__':

    computer_guesser = guessers.GensimGuesser()
    game = Game('resources/freqs.csv', 'resources/tokenized_100k_corp.spl', guesser=computer_guesser)
    game_lengths = [game.guessing_game(show_bert_output=True, full_sentence=False, number_of_subwords=i)
                    for i in [1, 2]]
    print(game_lengths)
