import os
import pickle
from collections import defaultdict
from typing import List, Iterable, Tuple
from itertools import islice, chain, repeat

import tqdm
import gensim
import numpy as np
import transformers
from torch import Tensor, where as torch_where, stack as torch_stack, prod as torch_prod
from torch.nn.functional import softmax

from utils.trie import Trie


class BertGuesser:

    def __init__(self, trie_fn: str = 'trie_words.pickle', wordlist_fn: str = 'resources/wordlist_3M.csv',
                 models_dir='models'):
        """
        Bert guesser class. Upon receiving a context, it returns guesses based on the Trie of available words.

        :param trie_fn: Filename for the tree. If the file is not available, it will be used as an output and a trie
        will be created at `trie_fn` location.
        :param wordlist_fn: If there is no trie supplemented, it will be created based on this file.
        :param models_dir: Thee directory where the models are stored.
        """

        self.tokenizer, self.model, self.word_trie = self.prepare_resources(trie_fn, wordlist_fn, models_dir)
        self.model.eval()  # makes output deterministic  # TODO Ezt nem lehetne letárolni?
        self.starting_words = {word_id: word for word, word_id in self.tokenizer.vocab.items() if word.isalpha()}
        self.center_words = {word_id: word for word, word_id in self.tokenizer.vocab.items() if word.startswith('##')}

    @staticmethod
    def prepare_resources(trie_fn: str, wordlist_fn: str, models_dir='models') -> Tuple:
        """
        Prepares resources by downloading and saving the transformer models and by creating (or loading) the word trie.

        :param trie_fn: Filename for the trie. If does not exist, will create it from `wordlist_fn`.
        :param wordlist_fn: Filename for the wordlist. Only used if the trie does not exist.
        :param models_dir: Thee directory where the models are stored.
        :return: A 3-long tuple containing a tokenizer, model and word trie.
        """
        if os.path.isdir(os.path.join(models_dir, 'hubert-base-cc')):
            tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(models_dir, 'hubert-base-cc'),
                                                                   lowercase=True)
            model = transformers.BertForMaskedLM.from_pretrained(os.path.join(models_dir, 'hubert-base-cc'),
                                                                 return_dict=True)
        else:
            # When first downloading the model, we save it, so we don't need internet later
            tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
            model = transformers.BertForMaskedLM.from_pretrained('SZTAKI-HLT/hubert-base-cc', return_dict=True)
            os.makedirs(models_dir, exist_ok=True)  # Do not fail when models directory exists
            tokenizer.save_pretrained(os.path.join(models_dir, 'hubert-base-cc'))
            model.save_pretrained(os.path.join(models_dir, 'hubert-base-cc'))

        # Create trie (assuming models_dir dir exists)
        trie_fn_path = os.path.join(models_dir, trie_fn)
        if os.path.isfile(trie_fn_path):
            with open(trie_fn_path, 'rb') as infile:
                word_trie: Trie = pickle.load(infile)
        else:
            print(f'Trie model file not found at {trie_fn} location, creating one from {wordlist_fn}.')
            word_trie = Trie()
            #  wordlist_tokenized always exists  # TODO ez mit jelent?
            with open(wordlist_fn) as infile:
                for line in tqdm.tqdm(infile, desc='Building trie... '):
                    line = line.rstrip()
                    if len(line) > 0:
                        word = tokenizer(line, add_special_tokens=False)['input_ids']
                        word_trie.insert(word)
            with open(trie_fn_path, 'wb') as outfile:
                pickle.dump(word_trie, outfile)
            print(f'Trie model created at {trie_fn_path} location.')

        return tokenizer, model, word_trie

    def _get_probabilities(self, masked_text: str) -> Tensor:
        """
        Creates the tensor from masked text.
        :param masked_text: A string containing the tokenizer's mask in text form. ([MASK])
        :return: The probability (softmax) tensor for the given [MASK] positions.
        """

        tokenized_text = self.tokenizer(masked_text, return_tensors='pt')
        mask_index = torch_where(tokenized_text['input_ids'][0] == self.tokenizer.mask_token_id)
        output = self.model(**tokenized_text)
        softmax_tensor = softmax(output.logits, dim=-1)
        # This is the probability vector for the masked WP's position
        probability_vector = softmax_tensor[0, mask_index[0], :]

        return probability_vector

    def make_guess(self, contexts: List[List[str]], word_length: int, number_of_subwords: int,
                   previous_guesses: Iterable[str], retry_wrong: bool = False, top_n: int = 10,
                   missing_token: str = 'MISSING') -> List[str]:
        """
        Main interface for the game. Processes list of words.

        :param contexts: contexts
        :param word_length: length of the missing word
        :param number_of_subwords: number of subwords to guess
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :param missing_token: String representation of the token marking the place of the guess.
        :return: BERT guesses in a list
        """

        softmax_tensors = self._calculate_softmax_from_context(contexts, number_of_subwords, missing_token)

        probability_tensor = torch_stack(softmax_tensors, dim=2)
        joint_probabilities = torch_prod(probability_tensor, dim=2)

        # TODO Ez mi?
        # length_combinations = self.knapsack_length(total_length=word_length, number_of_subwords=number_of_subwords)

        guesses = (guess for guess in self._softmax_iterator(joint_probabilities, target_word_length=word_length)
                   if retry_wrong or guess not in previous_guesses)

        # Pad to top_n if there is < top_n guesses present
        top_n_guesses = list(islice(chain(guesses, repeat('_')), top_n))

        return top_n_guesses

    def _calculate_softmax_from_context(self, contexts: List[List[str]], number_of_subwords: int,
                                        missing_token: str) -> List[Tensor]:
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
            unk_context = list(chain(context[:mask_loc], number_of_subwords * [self.tokenizer.mask_token],
                                     context[mask_loc + 1:]))
            bert_context = self.tokenizer(' '.join(unk_context))['input_ids']
            softmax_tensor = self._get_probabilities(self.tokenizer.decode(bert_context,
                                                                           clean_up_tokenization_spaces=True))
            softmax_tensors.append(softmax_tensor)

        return softmax_tensors

    def _softmax_iterator(self, probability_tensor: Tensor, target_word_length: int) -> str:

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
            candidates += [can for can, _ in self.word_trie.query_fixed_depth([word_id], length_subwords)]

        word_probabilities = np.prod(np.take(probabilities, candidates), axis=1).flatten()
        argsorted_probabilities = np.argsort(-word_probabilities)
        for idx in argsorted_probabilities:
            candidate = candidates[idx]
            word = self.tokenizer.decode(candidate, clean_up_tokenization_spaces=True)
            # TODO miért kell ellenőrizni a hosszt, ha a candidate már fix hosszú?
            if len(word) == target_word_length and word.isalpha():
                # Somehow BERT loves making multi-words with hyphens
                yield word


class GensimGuesser:
    def __init__(self, model_fn='hu_wv.gensim', models_dir='models'):
        self.model = gensim.models.Word2Vec.load(os.path.join(models_dir, model_fn))

    def make_guess(self, contexts: List[List[str]], word_length: int, number_of_subwords: int,
                   previous_guesses: Iterable[str], retry_wrong: bool, top_n: int = 10,
                   missing_token: str = 'MISSING') -> List[str]:
        """
        A gensim-based guesser. Since gensim's API is stable, it can be either FastText, CBOW or skip-gram, as long
        as the model has a `predict_output_word` method.
        Takes the top 1_000_000 guesses for each context, creates the intersection of these guesses, and returns
        the word with the highest possibility by taking the product of the possibilities for each context.

        :param contexts: contexts
        :param word_length: length of the missing word
        :param number_of_subwords: number of subwords to guess (not used here)
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :param missing_token: String representation of the token marking the place of the guess.
        :return: cbow guesses in a list
        """

        fixed_contexts = [[word for word in context if word != missing_token] for context in contexts]

        # We build the initial wordlist from the first guess
        # TODO 1_000_000 volt ígérve! Magyarázat?
        init_probabilities = {word: prob for word, prob in self.model.predict_output_word(fixed_contexts[0], 10_000)
                              if len(word) == word_length and (word not in previous_guesses or retry_wrong)}

        # And we filter it in the following guesses.
        probabilities = defaultdict(lambda x: float(1.0), init_probabilities)
        for context in fixed_contexts[1:]:
            output_dict = dict(self.model.predict_output_word(context, 10_000))
            for word in probabilities:
                if word in output_dict:
                    probabilities[word] *= output_dict[word]
                else:
                    probabilities[word] = 0

        # Sort results by descending probabilities
        words = list(islice((w for w, _ in sorted(probabilities.items(), key=lambda x: (-x[1], x[0]))), top_n))

        return words


def download():
    BertGuesser.prepare_resources('trie_words.pickle', 'resources/wordlist_3M.csv')


if __name__ == '__main__':
    # just download and build everything if the module is not imported
    download()
