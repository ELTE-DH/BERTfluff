import os
import transformers
from collections import Counter
import csv
import random
import torch
import torch.nn.functional as F
from typing import Generator, List, Iterable
import pickle
from utils.trie import Trie, TrieNode
import numpy as np


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


def tokenize_wpl_file(infilename: str, outfilename: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained('models/hubert-base-cc', lowercase=True)

    with open(infilename) as infile, open(outfilename, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for line in infile:
            input_ids = tokenizer(line, add_special_tokens=False)['input_ids']
            csv_writer.writerow([line.strip(), ' '.join(map(str, input_ids))])


def create_aligned_text(sentences: List[str]) -> List[str]:
    hashmark_positions = [sen.find('#') for sen in sentences]
    zero_point = max(hashmark_positions)
    return [' ' * (zero_point - position) + sentence for position, sentence in zip(hashmark_positions, sentences)]


def increment_list_index(lst: list, index) -> list:
    new_lst = lst.copy()
    new_lst[index] += 1
    return new_lst


def traverse_dims(lst: list, max_id: int) -> List[List]:
    candidates = []
    for i, elem in enumerate(lst):
        if elem < max_id:
            candidates.append(increment_list_index(lst, i))
        else:
            continue

    return candidates


def traverse_restricted(lst: list, max_ids: List[int]) -> List[List]:
    candidates = []
    for i, elem in enumerate(lst):
        if elem < max_ids[i]:
            candidates.append(increment_list_index(lst, i))
        else:
            continue
    return candidates


class Game:
    if 'models' in os.listdir('./'):
        tokenizer = transformers.AutoTokenizer.from_pretrained('models/hubert-base-cc', lowercase=True)
        model = transformers.BertForMaskedLM.from_pretrained('models/hubert-base-cc', return_dict=True)
    else:
        # if used with the online model, it will only start if internet is available due to checking the online cache
        # for a new model
        tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
        model = transformers.BertForMaskedLM.from_pretrained('SZTAKI-HLT/hubert-base-cc', return_dict=True)
        os.mkdir('models')
        tokenizer.save_pretrained('models/hubert-base-cc')
        model.save_pretrained('models/hubert-base-cc')

    def __init__(self, freqs_fn: str, corp_fn: str, word_list_fn: str):
        """

        :param freqs_fn: Word frequencies to choose.
        :param corp_fn: Corpus in SPL format.
        :param word_list_fn: File containing one word per line.
        :return:
        """
        self.counter = self.create_counter(filename=freqs_fn)
        self.corp_fn = corp_fn
        self.vocabulary = set()
        with open('models/trie_words.pickle', 'rb') as infile:
            self.word_trie: utils.trie.Trie = pickle.load(infile)

        with open(word_list_fn) as infile:
            for line in infile:
                self.vocabulary.add(line.strip().lower())

        # we create word lists for the starting/middle subwords
        # now we have to filter based on length

        self.starting_words = {id_: word for word, id_ in self.tokenizer.vocab.items() if word.isalpha()}
        self.center_words = {id_: word for word, id_ in self.tokenizer.vocab.items() if word[0:2] == '##'}

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

    def get_probabilities(self, masked_text: str) -> torch.Tensor:
        """
        Creates the tensor from masked text.
        :param masked_text:
        :return:
        """

        tokenized_text = self.tokenizer(masked_text, return_tensors='pt')
        mask_index = torch.where(tokenized_text['input_ids'][0] == self.tokenizer.mask_token_id)
        output = self.model(**tokenized_text)
        softmax = F.softmax(output.logits, dim=-1)
        probability_vector = softmax[0, mask_index[0], :]  # this is the probability vector for the masked WP's position
        return probability_vector

    def calculate_softmax_from_context(self, contexts: List[List[str]], number_of_subwords: int,
                                       missing_token: str = 'MISSING') -> List[torch.Tensor]:
        """
        Calculates the softmax tensors for a given lsit of contexts.
        :param missing_token: The name of the missing token.
        :param contexts: Multiple contexts, where one context is a list of strings, with the mask word being MISSING.
        :param number_of_subwords: Number of subwords in the `missing_token` location
        :return: Softmax tensors for each context
        """

        softmax_tensors = []
        for context in contexts:
            mask_loc = context.index(missing_token)
            unk_context = context[:mask_loc] + number_of_subwords * [self.tokenizer.mask_token] + context[mask_loc+1:]
            bert_context = self.tokenizer(' '.join(unk_context))['input_ids']
            softmax = self.get_probabilities(self.tokenizer.decode(bert_context))
            softmax_tensors.append(softmax)

        return softmax_tensors

    def calculate_guess(self, softmax_tensors: List[torch.Tensor], word_length: int,
                        previous_guesses: Iterable, retry_wrong: bool = False, top_n: int = 10) -> List[str]:

        probability_tensor = torch.stack(softmax_tensors, dim=2)
        joint_probabilities = torch.prod(probability_tensor, dim=2)

        # length_combinations = self.knapsack_length(total_length=word_length, number_of_subwords=number_of_subwords)

        guess_iterator = self.softmax_iterator(joint_probabilities, target_word_length=word_length)
        guesses = []
        for guess in guess_iterator:
            if retry_wrong:
                guesses.append(guess)
            else:
                if guess in previous_guesses:
                    continue
                else:
                    guesses.append(guess)
            if len(guesses) >= top_n:
                break

        return guesses

    def make_guess(self, contexts: List[List[str]], word_length: int, number_of_subwords: int,
                   previous_guesses: Iterable[str], retry_wrong: bool, top_n: int = 10) -> List[str]:
        """
        Main interface for the game. Processes list of words.
        :param contexts: contexts
        :param word_length: length of the missing word
        :param number_of_subwords: number of subwords to guess
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :return: BERT guesses in a list
        """

        softmax_tensors = self.calculate_softmax_from_context(contexts, number_of_subwords)
        guesses = self.calculate_guess(softmax_tensors, word_length, previous_guesses, retry_wrong, top_n)
        return guesses

    def softmax_iterator(self, probability_tensor: torch.Tensor, target_word_length: int) -> str:

        """
        Yields a valid guess (regardless of the previous guesses) by taking the joint probabilities and
        iterating over them in decreasing order of probability.
        The possible words are ordered into a trie,
        :param probability_tensor: Tensor containing joint probabilities
        :param target_word_length: Length of target word.
        :return: A guess with correct length and affixiation.
        """

        length_subwords = probability_tensor.shape[0]
        probabilities = probability_tensor.detach().numpy()
        candidates = []
        for word_id in self.starting_words:
            current_candidates = [can for can, _ in self.word_trie.query_fixed_depth([word_id], length_subwords)]
            candidates += current_candidates

        word_probabilities = np.prod(np.take(probabilities, candidates), axis=1).flatten()
        argsorted_probabilities = np.argsort(-word_probabilities)
        for idx in argsorted_probabilities:
            candidate = candidates[idx]
            word = self.tokenizer.decode(candidate)
            if len(word) == target_word_length and word.isalpha():
                # somehow BERT loves making multi-words with hyphens
                yield word

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

        print(len(selected_word), selected_wordids, self.counter[selected_word])

        for i, orig_sentence in enumerate(self.line_yielder(self.corp_fn, selected_word, full_sentence)):

            tokenized_sentence = orig_sentence.split(' ')
            mask_loc = tokenized_sentence.index(selected_word)

            masked_sentence = tokenized_sentence.copy()
            masked_sentence[mask_loc] = 'MISSING'

            contexts.append(masked_sentence)
            # softmax_tensors.append(self.get_probabilities(masked_sentence))
            bert_guesses = self.make_guess(contexts, word_length=len(selected_word),
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
                elif user_input.strip() == '':
                    user_guessed = True

            print(f'BERT\'s guess is {bert_guesses[:1]}')

            if show_bert_output:
                print(f'BERT\'s top 10 guesses: {" ".join(bert_guesses[:10])}')

            guess = bert_guesses[0] if len(bert_guesses) > 0 else ''

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
    game = Game('freqs.csv', 'tokenized_100k_corp.spl', 'wordlist_3M.csv')
    game_lengths = [game.guessing_game(show_bert_output=True, full_sentence=False, number_of_subwords=i) for i in range(1, 5)]
    print(game_lengths)
