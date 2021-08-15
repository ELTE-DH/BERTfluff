import os
import pickle
from typing import List, Iterable
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from utils.trie import Trie
import tqdm
import gensim
from collections import defaultdict


# needs trie node for the pickled object


class BertGuesser:
    if 'models' in os.listdir('./'):
        tokenizer = transformers.AutoTokenizer.from_pretrained('models/hubert-base-cc', lowercase=True)
        model = transformers.BertForMaskedLM.from_pretrained('models/hubert-base-cc', return_dict=True)
    else:
        # if used with the online model, it will only start if internet is available due to checking the online cache
        # for a new model, thus we save the model in `models` for later reuse
        tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
        model = transformers.BertForMaskedLM.from_pretrained('SZTAKI-HLT/hubert-base-cc', return_dict=True)
        if 'models' not in os.listdir('./'):
            os.mkdir('models')
        tokenizer.save_pretrained('models/hubert-base-cc')
        model.save_pretrained('models/hubert-base-cc')

    def __init__(self, trie_fn: str = 'trie_words.pickle', wordlist_fn: str = 'resources/wordlist_3M.csv'):
        """
        Bert guesser class. Upon receiving a context, it returns guesses based on the Trie of available words.
        :param trie_fn: Filename for the tree. If the file is not available, it will be used as an output and a trie
        will be created at `trie_fn` location.
        :param wordlist_fn: If there is no trie supplemented, it will be created based on this file.
        """

        self.word_trie = self.create_trie(trie_fn, wordlist_fn)
        self.starting_words = {id_: word for word, id_ in self.tokenizer.vocab.items() if word.isalpha()}
        self.center_words = {id_: word for word, id_ in self.tokenizer.vocab.items() if word[0:2] == '##'}

    def create_trie(self, trie_fn: str, wordlist_fn: str) -> Trie:

        if 'models' in os.listdir('./'):
            if trie_fn in os.listdir(f'models/'):
                with open(f'models/{trie_fn}', 'rb') as infile:
                    word_trie: Trie = pickle.load(infile)
            else:
                print(f'Trie model file not found at {trie_fn} location, creating one from {wordlist_fn}.')
                word_trie = Trie()
                #  wordlist_tokenized always exists
                with open(wordlist_fn) as infile:
                    for line in tqdm.tqdm(infile, desc='Building trie... '):
                        if len(line) <= 1 or len(line) >= 16:
                            continue
                        word = self.tokenizer(line.strip(), add_special_tokens=False)['input_ids']
                        word_trie.insert(word)
                with open(f'models/{trie_fn}', 'wb') as outfile:
                    pickle.dump(word_trie, outfile)
                print(f'Trie model created at models/{trie_fn} location.')

        return word_trie

    def get_probabilities(self, masked_text: str) -> torch.Tensor:
        """
        Creates the tensor from masked text.
        :param masked_text: A string containing the tokenizer's mask in text form. ([MASK])
        :return: The probability (softmax) tensor for the given [MASK] positions.
        """

        tokenized_text = self.tokenizer(masked_text, return_tensors='pt')
        mask_index = torch.where(tokenized_text['input_ids'][0] == self.tokenizer.mask_token_id)
        output = self.model(**tokenized_text)
        softmax = F.softmax(output.logits, dim=-1)
        probability_vector = softmax[0, mask_index[0], :]  # this is the probability vector for the masked WP's position
        return probability_vector

    def calculate_softmax_from_context(self, contexts: List[List[str]], number_of_subwords: int,
                                       missing_token: str) -> List[torch.Tensor]:
        """
        Calculates the softmax tensors for a given list of contexts.
        :param missing_token: The name of the missing token.
        :param contexts: Multiple contexts, where one context is a list of strings, with the mask word being MISSING.
        :param number_of_subwords: Number of subwords in the `missing_token` location
        :return: Softmax tensors for each context
        """

        softmax_tensors = []
        for context in contexts:
            mask_loc = context.index(missing_token)
            unk_context = context[:mask_loc] + number_of_subwords * [self.tokenizer.mask_token] + context[mask_loc + 1:]
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
            if retry_wrong or guess not in previous_guesses:
                guesses.append(guess)
            if len(guesses) >= top_n:
                break

        return guesses

    def make_guess(self, contexts: List[List[str]], word_length: int, number_of_subwords: int,
                   previous_guesses: Iterable[str], retry_wrong: bool, top_n: int = 10,
                   missing_token: str = 'MISSING') -> List[str]:

        """
        Main interface for the game. Processes list of words.

        :param missing_token: String representation of the token marking the place of the guess.
        :param contexts: contexts
        :param word_length: length of the missing word
        :param number_of_subwords: number of subwords to guess
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :return: BERT guesses in a list
        """

        softmax_tensors = self.calculate_softmax_from_context(contexts, number_of_subwords, missing_token)
        guesses = self.calculate_guess(softmax_tensors, word_length, previous_guesses, retry_wrong, top_n)
        return guesses

    def softmax_iterator(self, probability_tensor: torch.Tensor, target_word_length: int) -> str:

        """
        Yields a valid guess (regardless of the previous guesses) by taking the joint probabilities and
        iterating over them in decreasing order of probability.
        The possible words are ordered into a trie.

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


class GensimGuesser:
    def __init__(self, model_fn='models/hu_wv.gensim'):
        self.model = gensim.models.Word2Vec.load(model_fn)

    @staticmethod
    def create_compatible_context(context: List[str], missing_token: str) -> List[str]:

        return [word for word in context if word != missing_token]

    def make_guess(self, contexts: List[List[str]], word_length: int, number_of_subwords: int,
                   previous_guesses: Iterable[str], retry_wrong: bool, top_n: int = 10,
                   missing_token: str = 'MISSING') -> List[str]:
        """
        A gensim-based guesser. Since gensim's API is stable, it can be either FastText, CBOW or skip-gram, as long
        as the model has a `predict_output_word` method.
        Takes the top 1_000_000 guesses for each context, creates the intersection of these guesses, and returns
        the word with the highest possibility by taking the product of the possibilities for each context.

        :param missing_token: String representation of the token marking the place of the guess.
        :param contexts: contexts
        :param word_length: length of the missing word
        :param number_of_subwords: number of subwords to guess, it is unused in this method
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :return: guesses in a list
        """

        fixed_contexts = [self.create_compatible_context(context, missing_token) for context in contexts]
        probabilities = defaultdict(lambda: 1.0)

        first_round = True
        for context in fixed_contexts:
            for word, prob in self.model.predict_output_word(context, 1_000_000):
                # The only correct words are
                if len(word) == word_length and (word in probabilities or first_round) and (
                        word not in previous_guesses or retry_wrong):
                    probabilities[word] *= prob
            first_round = False

        retval = [word for word, _ in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)][:top_n]

        return retval


class DummyGuesser:
    """
    Dummy guesser. Provides the make_guess method in case of another guesser fails.
    """

    @staticmethod
    def make_guess(top_n: int = 10, *_, **__) -> List[str]:
        return ['_'] * top_n
