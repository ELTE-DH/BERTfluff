
from typing import List, Tuple

from helper import GensimHelper
from guessers import BertGuesser
from context_bank import ContextBank


class Game:
    def __init__(self, freqs_fn: str, corp_fn: str, guesser, sim_helper: GensimHelper = None):
        self.context_bank = ContextBank(freqs_fn, corp_fn)
        self.guesser = guesser
        self.similarity_helper = sim_helper

    @staticmethod
    def _create_aligned_text(sentences: List[str]) -> List[str]:
        hashmark_positions = [sen.find('#') for sen in sentences]
        zero_point = max(hashmark_positions)
        return [' ' * (zero_point - position) + sentence for position, sentence in zip(hashmark_positions, sentences)]

    def _user_experience(self, selected_word: str, sentences: List[str], user_guessed: bool,
                         computer_guessed: bool, show_model_output: bool,
                         computer_guesses: List[str]) -> Tuple[bool, bool]:
        """
        Provides the user experience.

        :param selected_word: The word missing from the text.
        :param sentences: Previous sentence already masked with hashmarks.
        :param user_guessed: Whether the user guessed already or not. If did, then it skips over the user-specific
                              questions.
        :param computer_guessed: Whether the computer guessed correctly or not. If yes, skips over
                                  the computer-specific part.
        :param show_model_output: If its on, it prints the top 10 guesses of the computer.
        :param computer_guesses:
        :return: Length of game for the player and the computer
        """

        print('\n'.join(self._create_aligned_text(sentences)))
        print('-' * 80)

        if not user_guessed:
            user_input = input('Please input your guess: ').strip()
            if self.similarity_helper and user_input:
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

        print(f'Computer\'s guess is {computer_guesses[0]}')

        if show_model_output:
            print(f'Computer\'s top 10 guesses: {" ".join(computer_guesses[:10])}')

        computer_guess = computer_guesses[0] if len(computer_guesses) > 0 else ''

        if computer_guess == '_' or len(sentences) > 10:  # if the computer takes too long to figure it out
            print('Computer gave up.')
            computer_guessed = True

        if not computer_guessed and computer_guess != '_':
            computer_guessed = computer_guess == selected_word
            if computer_guessed:
                print('Computer guessed the word.')

        return user_guessed, computer_guessed

    def guessing_game(self, show_model_output: bool = True, full_sentence: bool = False,
                      number_of_subwords: int = 1, debug: bool = False) -> dict:
        """
        Provides the interface for the game.

        :return: a dictionary of length 3, containing the number of guesses of the player, BERT and the word missing
        """
        selected_word, selected_wordids, selected_word_freq = self.context_bank.select_word(number_of_subwords)

        computer_history = set()
        user_guessed = False
        computer_guessed = False
        retval = {'user_attempts': -1, 'computer_attempts': -1, 'missing_word': selected_word}
        human_contexts = []
        computer_contexts = []

        if debug:
            print(selected_word)

        print(len(selected_word), selected_wordids, selected_word_freq)

        for i, (original_sentence, (computer_masked_sentence, hashmarked_sentence)) \
                in enumerate(self.context_bank.line_yielder(selected_word, full_sentence)):

            human_contexts.append(hashmarked_sentence)
            computer_contexts.append(computer_masked_sentence)
            computer_current_guesses = self.guesser.make_guess(computer_contexts, word_length=len(selected_word),
                                                               previous_guesses=computer_history,
                                                               retry_wrong=False,
                                                               number_of_subwords=len(selected_wordids))

            computer_history.add(computer_current_guesses[0])

            user_guessed, computer_guessed = self._user_experience(selected_word, human_contexts, user_guessed,
                                                                   computer_guessed, show_model_output,
                                                                   computer_current_guesses)

            # We log how many guesses it took for the players.
            if user_guessed and retval['user_attempts'] == -1:
                retval['user_attempts'] = i + 1
            if computer_guessed and retval['computer_attempts'] == -1:
                retval['computer_attempts'] = i + 1

            if computer_guessed and user_guessed:
                return retval

        # in case player does not guess it, we return
        return retval


if __name__ == '__main__':
    computer_guesser = BertGuesser()
    print('Guesser loaded!')
    guess_helper = GensimHelper()
    print('Helper loaded!')
    game = Game('resources/freqs.csv', 'resources/tokenized_100k_corp.spl', guesser=computer_guesser,
                sim_helper=guess_helper)
    print('Game handler loaded!')
    game_lengths = [game.guessing_game(number_of_subwords=i)  # show_model_output=True, full_sentence=False,
                    for i in (1, 1, 2)]
    print(game_lengths)
