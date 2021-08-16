import csv
import random
from collections import Counter
from typing import Generator, List, Tuple
import transformers
import guessers
import helper


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
    def __init__(self, freqs_fn: str, corp_fn: str, guesser, sim_helper: helper.GensimHelper = None):
        """

        :param freqs_fn: Word frequencies to choose.
        :param corp_fn: Corpus in SPL format.
        :return:
        """
        self.counter = self.create_counter(filename=freqs_fn)
        self.corp_fn = corp_fn
        self.guesser = guesser
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
        self.similarity_helper = sim_helper

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

    def select_word(self, number_of_subwords: int) -> Tuple[str, list]:
        """
        Selects a word from the self.counter dictionary.

        :param number_of_subwords:
        :return:
        """

        selected_word, selected_input_ids = '', []
        while len(selected_input_ids) != number_of_subwords:
            selected_word = random.choice(list(self.counter.keys()))
            selected_input_ids = self.tokenizer(selected_word, add_special_tokens=False)['input_ids']

        return selected_word, selected_input_ids

    def user_experience(self, selected_word: str, sentences: List[str], user_guessed: bool,
                        computer_guessed: bool, show_model_output: bool, computer_guesses: List[str]):
        """
        Provides the user experience.

        :param selected_word: The word missing from the text.
        :param sentences: Previous sentence already masked with hashmarks.
        :param user_guessed: Whether the user guessed already or not. If did, then it skips over the user-specific questions.
        :param computer_guessed: Whether the computer guessed correctly or not. If yes, skips over the computer-specific part.
        :param show_model_output: If its on, it prints the top 10 guesses of the computer.
        :param computer_guesses:
        :return:
        """

        print('\n'.join(create_aligned_text(sentences)))
        print('-' * 80)

        if not user_guessed:
            user_input = input('Please input your guess: ').strip()
            if self.similarity_helper:
                # if there is a similarity helper
                similarity = self.similarity_helper.word_similarity(selected_word, user_input)
                # similarity is -1 by either the model not containing one of the words
                # in case of missing words, the function prints the word missing
                if similarity != -1.0:
                    print(f'Your guess has a {similarity:.3f} similarity to the missing word.')
            if user_input == '':
                #  if the user gives up, we let go of them -- by setting user guessed to true
                user_guessed = True
                print(f'You gave up.')

            if not user_guessed and user_input == selected_word:
                user_guessed = True
                print(f'Your guess ({user_input}) was correct.')

        print(f'Computer\'s guess is {"".join(computer_guesses[:1])}')

        if show_model_output:
            print(f'Computer\'s top 10 guesses: {" ".join(computer_guesses[:10])}')

        computer_guess = computer_guesses[0] if len(computer_guesses) > 0 else ''

        if not computer_guessed:
            computer_guessed = computer_guess == selected_word
            if computer_guessed:
                print('Computer guessed the word.')

        return user_guessed, computer_guessed

    @staticmethod
    def mask_sentence(original_sentence: str, missing_word: str) -> Tuple[List[str], str]:
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

    def guessing_game(self, show_model_output: bool = True, full_sentence: bool = False,
                      number_of_subwords: int = 1) -> dict:
        """
        Provides the interface for the game.
        :return: a dictionary of length 3, containing the number of guesses of the player, BERT and the word missing
        """
        selected_word, selected_wordids = self.select_word(number_of_subwords)

        computer_history = set()
        user_guessed = False
        computer_guessed = False
        retval = {'user_attempts': -1, 'computer_attempts': -1, 'missing_word': selected_word}
        human_contexts = []
        computer_contexts = []

        # print(selected_word)
        print(len(selected_word), selected_wordids, self.counter[selected_word])

        for i, original_sentence in enumerate(self.line_yielder(self.corp_fn, selected_word, full_sentence)):

            computer_masked_sentence, hashmarked_sentence = self.mask_sentence(original_sentence, selected_word)
            human_contexts.append(hashmarked_sentence)
            computer_contexts.append(computer_masked_sentence)
            computer_current_guesses = self.guesser.make_guess(computer_contexts, word_length=len(selected_word),
                                                               previous_guesses=computer_history,
                                                               retry_wrong=False,
                                                               number_of_subwords=len(selected_wordids))

            computer_history.add(computer_current_guesses[0])

            user_guessed, computer_guessed = self.user_experience(selected_word, human_contexts, user_guessed,
                                                                  computer_guessed, show_model_output,
                                                                  computer_current_guesses)

            # We log how many guesses it took for the players.
            if user_guessed and retval['user_attempts'] == -1:
                retval['user_attempts'] = i+1
            if computer_guessed and retval['computer_attempts'] == -1:
                retval['computer_attempts'] = i+1

            if computer_guessed and user_guessed:
                return retval

        # in case player does not guess it, we return
        return retval


if __name__ == '__main__':

    computer_guesser = guessers.BertGuesser()
    guess_helper = helper.GensimHelper()
    game = Game('resources/freqs.csv', 'resources/tokenized_100k_corp.spl', guesser=computer_guesser,
                sim_helper=guess_helper)
    game_lengths = [game.guessing_game(show_model_output=True, full_sentence=False, number_of_subwords=i)
                    for i in [1, 2]]
    print(game_lengths)
