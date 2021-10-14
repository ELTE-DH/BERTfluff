
from typing import List, Tuple

from tabulate import tabulate

from helper import GensimHelper
from guessers import BertGuesser
from context_bank import ContextBank
from context_bank_sql import ContextBank as ContextBankSQL


class Game:
    def __init__(self, context_bank: ContextBank, guesser, sim_helper: GensimHelper = None):
        self.context_bank = context_bank
        self.guesser = guesser
        self.similarity_helper = sim_helper

    @staticmethod
    def _print_aligned_sentences(sentences: List[Tuple[str, str, str]]):
        hashmark_positions = [len(left) + len(word) + 2 for left, word, _ in sentences]
        zero_point = max(hashmark_positions)
        aligned_stencences = [f'{" " * (zero_point - position)}{left} {word} {right}'
                              for position, (left, word, right) in zip(hashmark_positions, sentences)]
        print('\n'.join(aligned_stencences))
        print('-' * 80)

    def _user_experience(self, selected_word: str, sentences: List[Tuple[str, str, str]], user_guessed: bool,
                         user_gave_up: bool, computer_guessed: bool, computer_gave_up: bool, show_model_output: bool,
                         computer_guesses: List[str]) -> Tuple[bool, bool, bool, bool]:
        """
        Provides the user experience.

        :param selected_word: The word missing from the text.
        :param sentences: Previous sentence already masked with hashmarks.
        :param user_guessed: Whether the user guessed already or not. If did, then it skips over the user-specific
                              questions.
        :param user_gave_up: Whether the user gave up already or not. If did, then it skips over the user-specific
                              questions.
        :param computer_guessed: Whether the computer guessed correctly or not. If yes, skips over
                                  the computer-specific part.
        :param computer_gave_up: Whether the computer gave up or not. If yes, skips over the computer-specific part.
        :param show_model_output: If its on, it prints the top 10 guesses of the computer.
        :param computer_guesses:
        :return: Length of game for the player and the computer
        """

        self._print_aligned_sentences(sentences)

        if not (user_guessed or user_gave_up):
            user_input = input('Please input your guess: ').strip()

            if self.similarity_helper and user_input:
                # If there is a similarity helper
                similarity = self.similarity_helper.word_similarity(selected_word, user_input)
                # Similarity is -1 if the model not containing either the guessed word or the missing word
                if similarity > -1.0:
                    print(f'Your guess has a {similarity:.3f} similarity to the missing word (the higher, the better).')

            if user_input == selected_word:
                user_guessed = True
                print(f'Your guess ({user_input}) was correct.')
            elif user_input == '':
                #  If the user gives up, we let go of them -- by setting user guessed to true
                user_gave_up = True
                print(f'You gave up.')

        computer_guess = computer_guesses[0] if len(computer_guesses) > 0 else '_'

        print(f'Computer\'s guess is {computer_guess}')

        if show_model_output:
            print(f'Computer\'s top 10 guesses: {" ".join(computer_guesses[:10])}')

        if computer_guess == selected_word:
            print('Computer guessed the word.')
            computer_guessed = True
        elif computer_guess == '_' or len(sentences) > 10:
            # If the computer gave up or would take too long to figure it out
            print('Computer gave up.')
            computer_gave_up = True

        print()

        return user_guessed, user_gave_up, computer_guessed, computer_gave_up

    def guessing_game(self, show_model_output: bool = True, context_size: int = 5,
                      number_of_subwords: int = 1, debug: bool = False) -> dict:
        """
        Provides the interface for the game.

        :return: a dictionary containing the results
        """

        selected_word, selected_word_freq, selected_wordids = '', 0, []
        while len(selected_wordids) != number_of_subwords:
            selected_word, selected_word_freq = self.context_bank.select_random_word()
            selected_wordids = self.guesser.split_to_subwords(selected_word)

        selected_word_len, selected_wordids_len = len(selected_word), len(selected_wordids)

        if debug:
            print(f'Selected word: {selected_word}')

        print(f'Selected word length: {selected_word_len}')
        print(f'Selected word word_ids: {selected_wordids}')
        print(f'Selected word freq: {selected_word_freq}')

        user_guessed, user_gave_up, computer_guessed, computer_gave_up = False, False, False, False
        retval = {'user_attempts': 0, 'computer_attempts': 0, 'missing_word': selected_word}
        computer_history, contexts, = [], []

        _, new_lines = self.context_bank.read_all_lines_for_word(selected_word, [], context_size)
        for _, left, hidden_word, right in new_lines:

            contexts.append((left, hidden_word, right))
            computer_current_guesses = self.guesser.make_guess(contexts, previous_guesses=computer_history,
                                                               retry_wrong=False,
                                                               number_of_subwords=selected_wordids_len)

            computer_history.append(computer_current_guesses[0])

            user_guessed, user_gave_up, computer_guessed, computer_gave_up = \
                self._user_experience(selected_word, contexts, user_guessed, user_gave_up, computer_guessed,
                                      computer_gave_up, show_model_output, computer_current_guesses)

            # We log how many guesses it took for the players before wining or giving up
            retval['user_attempts'] += int(not (user_guessed or user_gave_up))
            retval['computer_attempts'] += int(not (computer_guessed or computer_gave_up))

            # Game ends if the computer has guessed the word (reveals the solution) or
            #  user guessed the word -> we let the computer try further (guess or give up)
            #  user gave up -> we let the computer try further (guess or give up)
            if computer_guessed or (user_guessed and computer_gave_up) or (user_gave_up and computer_gave_up):
                # Add attempts of guessing the right word (only once) if it were guessed
                retval['user_attempts'] += int(user_guessed)
                retval['computer_attempts'] += int(computer_guessed)
                # Compute winner
                # User guessed the right word before the computer or the computer gave up when the user guesser right
                retval['user_won'] = user_guessed and \
                    (retval['user_attempts'] < retval['computer_attempts'] or computer_gave_up)
                # The computer guessed the right word before the user (fewer attempts or the user gave up)
                retval['computer_won'] = computer_guessed and not user_guessed
                # Both guessed in the same round or both gave up
                retval['tie'] = (user_gave_up and computer_gave_up) or \
                    (user_guessed and computer_guessed and retval['user_attempts'] == retval['computer_attempts'])

                break  # End game

        return retval


def main():
    context_bank = ContextBank('freqs.csv', 'tokenized_100k_corp.spl')
    # db_config = {'database_name': 'webcorpus_conc.db', 'table_name': 'lines', 'id_name': 'id','left_name': 'left',
    #              'word_name': 'word', 'right_name': 'right', 'freq': 'freq'}
    # context_bank = ContextBankSQL(db_config)
    print('Context Bank loaded!')
    computer_guesser = BertGuesser()
    print('Guesser loaded!')
    guess_helper = GensimHelper()
    print('Helper loaded!')
    game = Game(context_bank, computer_guesser, sim_helper=guess_helper)
    print('Game handler loaded!')

    table = []
    headers = ['missing_word', 'user_won', 'computer_won', 'tie', 'user_attempts', 'computer_attempts']
    for i in (1, 1, 2):
        result = game.guessing_game(number_of_subwords=i)  # show_model_output=True, full_sentence=False
        table.append([result[key] for key in headers])

    print('', 'Results:', sep='\n')
    print(tabulate(table, headers, tablefmt='github'))


if __name__ == '__main__':
    main()
