import transformers
from collections import Counter
import csv
import random
import torch
import torch.nn.functional as F
from typing import Generator, List, Iterable
from collections import defaultdict
from itertools import product


# suppress warning, only complain when halting
# transformers.logging.set_verbosity_error()


def create_corpora():
    """
    Used to create frequency. It also deduplicates in a rudimentary manner.
    """
    c = Counter()
    sentences = set()
    dupes = 0
    with open('100k_tok.spl') as infile, open('tokenized_100k_corp.spl', 'w') as outfile:
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

    with open('freqs.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for word, freq in sorted(c.items(), key=lambda x: x[1], reverse=True):
            csv_writer.writerow([word, freq])


def create_aligned_text(sentences: List[str]) -> List[str]:
    hashmark_positions = [sen.find('#') for sen in sentences]
    zero_point = max(hashmark_positions)
    return [' ' * (zero_point - position) + sentence for position, sentence in zip(hashmark_positions, sentences)]


class Game:
    # put them into class variable so in case of a REST API, multiple games can share the memory-hungry model
    tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
    model = transformers.BertForMaskedLM.from_pretrained('SZTAKI-HLT/hubert-base-cc', return_dict=True)

    def __init__(self, freqs_fn: str, corp_fn: str):
        """

        :param freqs_fn:
        :param corp_fn:
        :return:
        """
        self.counter = self.create_counter(filename=freqs_fn)
        self.corp_fn = corp_fn

        # we create word lists for the starting/middle subwords
        # now we have to filter based on length
        self.starting_words_by_length = defaultdict(lambda: defaultdict(dict))
        self.center_words_by_length = defaultdict(lambda: defaultdict(dict))

        for word, id_ in self.tokenizer.vocab.items():
            if word[0:2] == '##':
                self.center_words_by_length[len(word)-2][id_] = word[2:]
            elif word.isalpha():
                self.starting_words_by_length[len(word)][id_] = word
            else:
                continue


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

    def line_yielder(self, fname: str, word: str, full_sentence: bool) -> Generator:
        """
        With a word_id, it starts tokenizing the corpora (which is fast, hence not precomputed),
        and when finds a sentence containing the token, it yields the sentence.
        :param fname: corpus file in spl format
        :param word:
        :param full_sentence:
        :return:
        """
        with open(fname) as f:
            for line in f:
                sentence = line.strip().split(' ')
                if word in sentence:
                    if full_sentence:
                        yield line.strip()
                    else:
                        yield self.create_context(sentence, target_word=word)
                else:
                    continue

    def topn_fixed_length(self, mask_word: torch.Tensor, top_n: int, word_length: int,
                          check_word_length: bool = True) -> List[str]:
        """
        :param mask_word: 1-dim softmax tensor containing the probability for each word in the tokenizer's vocab
        :param top_n:
        :param word_length:
        :param check_word_length:
        :return: list of words in decreasing order of probability
        """
        order = torch.argsort(mask_word, dim=1, descending=True)[0]
        possible_words = []
        for token_id in order:
            token = self.tokenizer.decode([token_id])
            if check_word_length:
                if len(token) == word_length:
                    possible_words.append(token)
            else:
                possible_words.append(token)
            if len(possible_words) == top_n:
                break
        return possible_words

    @staticmethod
    def create_context(sentence: List[str], target_word: str, window_size: int = 3) -> str:
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
        # new_sentence = self.tokenizer.decode(sentence).split(' ')
        # return ' '.join(new_sentence[max(0, center-window_size):min(len(sentence), center+window_size+1)])

    def get_probabilities(self, masked_text: str) -> torch.Tensor:
        """
        Creates the probability vector given the masked text.
        :param masked_text:
        :return: Probability vector
        """

        tokenized_text = self.tokenizer(masked_text, return_tensors='pt')
        mask_index = torch.where(tokenized_text['input_ids'][0] == self.tokenizer.mask_token_id)
        output = self.model(**tokenized_text)
        softmax = F.softmax(output.logits, dim=-1)
        probability_vector = softmax[0, mask_index, :]  # this is the probability vector for the masked WP's position
        return probability_vector

    def get_multi_probabilities(self, masked_text: str) -> torch.Tensor:

        tokenized_text = self.tokenizer(masked_text, return_tensors='pt')
        mask_index = torch.where(tokenized_text['input_ids'][0] == self.tokenizer.mask_token_id)
        output = self.model(**tokenized_text)
        softmax = F.softmax(output.logits, dim=-1)
        probability_vector = softmax[0, mask_index[0], :]  # this is the probability vector for the masked WP's position
        return probability_vector

    def make_multi_guess(self, softmax_tensors: List[torch.Tensor], word_length: int,
                         previous_guesses: Iterable, retry_wrong: bool = False,
                         number_of_subwords: int = None) -> List[List[str]]:

        probability_tensor = torch.stack(softmax_tensors, dim=2)
        joint_probabilities = torch.prod(probability_tensor, dim=2)

        length_combinations = self.knapsack_length(total_length=word_length, number_of_subwords=number_of_subwords)

        multiwords = self.iterate_multiwords(length_combinations)

        # This approach will not work since multiwords grows exponentially
        # But in case of multiple guesses, we can iterate on the elementwise product of SUBWORD * 32000 size tensor
        # and check for lengths. It will find a candidate quick - perhaps.
        return [['alma', 'alma'], ['korte', 'korte']]

    def iterate_multiwords(self, length_combinations: List[List[int]]):

        retval = []
        for combination in length_combinations: # first is the length of the starting word, the rest are the rest
            for starting_word_id in self.starting_words_by_length[combination[0]]:
                center_lengths = combination[1:]
                if len(center_lengths) == 0:
                    retval.append((starting_word_id,))
                    continue
                else:
                    center_iterator = [sorted(self.center_words_by_length[length].keys()) for length in center_lengths]
                for multiword_idx in product(*center_iterator):
                    retval.append((starting_word_id, *multiword_idx))

        return retval

    def knapsack_length(self, total_length: int, number_of_subwords: int) -> List[List[int]]:
        """
        Calculates possible lengths for a given length of word. E.g. if the word is made of 10 chars and 2 subwords,
        it returns a the list [1,9], [2, 8] etc.
        :param total_length:
        :param number_of_subwords:
        :return:
        """
        combinations = [sorted(self.starting_words_by_length.keys())]
        for _ in range(number_of_subwords - 1):
            combinations.append(sorted(self.center_words_by_length.keys()))

        possible_combinations = [c for c in product(*combinations) if sum(c) == total_length]

        return possible_combinations

    def make_guess(self, softmax_tensors: List[torch.Tensor], word_length: int,
                   previous_guesses: Iterable[str], retry_wrong: bool = False) -> List[str]:

        """
        Makes a guess based on the previous guesses, maximizing the joint probability.
        :param softmax_tensors: Previous softmax tensors
        :param word_length: Length of the missing word
        :param previous_guesses: Previous guesses
        :param retry_wrong: Whether to retry the previous, bad guesses
        :return: List of guesses
        """

        # we stack the probability tensors on top of each other and select the word with the biggest product along
        # the guess axis (which is dim=2)
        probability_tensor = torch.stack(softmax_tensors, dim=2)
        joint_probabilities = torch.prod(probability_tensor, dim=2)

        top_n = self.topn_fixed_length(joint_probabilities, 20, word_length=word_length, check_word_length=True)
        if not retry_wrong:
            return [word for word in top_n if word not in previous_guesses]
        else:
            return top_n

    def make_turn(self, selected_wordid: int, user_guessed: bool, bert_guessed: bool, previous_guesses: Iterable[str],
                  previous_sentences: Iterable[str], softmax_tensors: List[List[float]]):
        """
        Stateless turn-maker for the REST API. WIP.
        :param selected_wordid:
        :param user_guessed:
        :param bert_guessed:
        :param previous_guesses:
        :param previous_sentences:
        :param softmax_tensors:
        :return:
        """

        selected_word = self.tokenizer.decode([selected_wordid])

    def guessing_game(self, show_bert_output: bool = True, full_sentence: bool = False) -> List:
        """
        Provides the interface for the game.
        :return: a list of length 3, containing the number of guesses of the player, BERT and the word missing
        """
        while True:
            selected_word = random.choice(list(self.counter.keys()))
            selected_wordids = self.tokenizer(selected_word, add_special_tokens=False)
            if len(selected_wordids['input_ids']) != 1:
                break
        guesses = set()
        user_guessed = False
        bert_guessed = False
        retval = [-1, -1, selected_word]
        softmax_tensors = []
        sentences = []

        print(len(selected_word), selected_wordids, self.counter[selected_word])

        for i, orig_sentence in enumerate(self.line_yielder(self.corp_fn, selected_word, full_sentence)):

            bert_sentence = self.tokenizer(orig_sentence)['input_ids']
            masked_sentence = bert_sentence.copy()
            for wordid in selected_wordids['input_ids']:
                masked_sentence[masked_sentence.index(wordid)] = self.tokenizer.mask_token_id

            # for multi-word prediction, the next step is to create a function which constructs whole words with a
            # given length
            # we have to iterate over the Descartes-product of the softmaxes, with some conditions:
            # sum of lengths must be equal to the original word's length
            # Only the first token is "beginning-type", the others start with ##
            # Compute everything first then filtering cuts back on the code footprint
            # While filtering, then cutting cuts back on the compute and memory footprint
            #
            masked_sentence_text = self.tokenizer.decode(masked_sentence)
            softmax = self.get_multi_probabilities(masked_sentence_text)
            softmax_tensors.append(softmax)
            # softmax_tensors.append(self.get_probabilities(masked_sentence))
            bert_guesses = self.make_multi_guess(softmax_tensors, word_length=len(selected_word),
                                                 previous_guesses=guesses, retry_wrong=False,
                                                 number_of_subwords=len(selected_wordids['input_ids']))
            # bert_guesses = self.make_guess(softmax_tensors, word_length=len(selected_word),
            #                                previous_guesses=guesses, retry_wrong=False)

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

            print(f'BERT\'s guess is {bert_guesses[0]}')

            if show_bert_output:
                print(f'BERT\'s top 5 guesses: {" ".join(bert_guesses[:5])}')
            guess = bert_guesses[0]

            if not bert_guessed:
                if guess == selected_word:
                    print('BERT guessed the word.')
                    bert_guessed = True
                    retval[1] = i + 1
                else:
                    guesses.add(guess)

            if bert_guessed and user_guessed:
                return retval

        # in case player does not guess it, we return
        return retval


if __name__ == '__main__':
    game = Game('freqs.csv', 'tokenized_100k_corp.spl')
    game_lengths = [game.guessing_game(show_bert_output=False, full_sentence=False) for i in range(5)]
    print(game_lengths)
