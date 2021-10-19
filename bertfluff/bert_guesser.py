import pickle
from typing import List, Tuple
from os import makedirs as os_makedirs
from itertools import islice, chain, repeat
from os.path import join as os_path_join, isdir as os_path_isdir, isfile as os_path_isfile

import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import AutoTokenizer, BertForMaskedLM
from torch import Tensor, where as torch_where, stack as torch_stack, prod as torch_prod

from bertfluff.trie import Trie


# suppress warning, only complain when halting
# transformers.logging.set_verbosity_error()


class BertGuesser:

    def __init__(self, trie_fn: str = 'trie_words.pickle', wordlist_fn: str = 'wordlist_3M.csv',
                 resources_dir: str = 'resources',  models_dir: str = 'models'):
        """
        Bert guesser class. Upon receiving a context, it returns guesses based on the Trie of available words.

        :param trie_fn: Filename for the tree. If the file is not available, it will be used as an output and a trie
        will be created at `trie_fn` location.
        :param wordlist_fn: If there is no trie supplemented, it will be created based on this file.
        :param resources_dir: The directory where the resources are stored.
        :param models_dir: The directory where the models are stored.
        """

        self.tokenizer, self.model, self.word_trie = \
            self.prepare_resources(trie_fn, wordlist_fn, resources_dir, models_dir)
        self.model.eval()  # Makes output deterministic
        self.starting_words = {word_id: word for word, word_id in self.tokenizer.vocab.items() if word.isalpha()}
        self.center_words = {word_id: word for word, word_id in self.tokenizer.vocab.items() if word.startswith('##')}
        self.tokenizer = AutoTokenizer.from_pretrained(os_path_join(models_dir, 'hubert-base-cc'), lowercase=True)

    @staticmethod
    def prepare_resources(trie_fn: str, wordlist_fn: str, resources_dir: str = 'resources',
                          models_dir: str = 'models') -> Tuple:
        """
        Prepares resources by downloading and saving the transformer models and by creating (or loading) the word trie.

        :param trie_fn: Filename for the trie. If does not exist, will create it from `wordlist_fn`.
        :param wordlist_fn: Filename for the wordlist. Only used if the trie does not exist.
        :param resources_dir: The directory where the resources are stored.
        :param models_dir: The directory where the models are stored.
        :return: A 3-long tuple containing a tokenizer, model and word trie.
        """
        if os_path_isdir(os_path_join(models_dir, 'hubert-base-cc')):
            tokenizer = AutoTokenizer.from_pretrained(os_path_join(models_dir, 'hubert-base-cc'), lowercase=True)
            model = BertForMaskedLM.from_pretrained(os_path_join(models_dir, 'hubert-base-cc'), return_dict=True)
        else:
            # When first downloading the model, we save it, so we don't need internet later
            tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
            model = BertForMaskedLM.from_pretrained('SZTAKI-HLT/hubert-base-cc', return_dict=True)
            os_makedirs(models_dir, exist_ok=True)  # Do not fail when models directory exists
            tokenizer.save_pretrained(os_path_join(models_dir, 'hubert-base-cc'))
            model.save_pretrained(os_path_join(models_dir, 'hubert-base-cc'))

        # Create trie (assuming models_dir dir exists)
        trie_fn_path = os_path_join(models_dir, trie_fn)
        if os_path_isfile(trie_fn_path):
            with open(trie_fn_path, 'rb') as infile:
                word_trie: Trie = pickle.load(infile)
        else:
            wordlist_fn_path = os_path_join(resources_dir, wordlist_fn)
            print(f'Trie model file not found at {trie_fn_path} location, creating one from {wordlist_fn_path}.')
            word_trie = Trie()
            #  wordlist_tokenized always exists  # TODO ez mit jelent?
            with open(wordlist_fn_path) as infile:
                for line in tqdm(infile, desc='Building trie... '):
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

    def make_guess(self, contexts: List[Tuple[str, str, str]], number_of_subwords: int,
                   previous_guesses: List[str], retry_wrong: bool = False, top_n: int = 10) -> List[str]:
        """
        Main interface for the game. Processes list of words.

        :param contexts: contexts
        :param number_of_subwords: number of subwords to guess
        :param previous_guesses: previous guesses
        :param retry_wrong: whether to retry or discard previous guesses
        :param top_n: number of guesses
        :return: BERT guesses in a list
        """

        previous_guesses = set(previous_guesses)
        softmax_tensors = self._calculate_softmax_from_context(contexts, number_of_subwords)

        probability_tensor = torch_stack(softmax_tensors, dim=2)
        joint_probabilities = torch_prod(probability_tensor, dim=2)

        # TODO Ez mi?
        # length_combinations = self.knapsack_length(total_length=word_length, number_of_subwords=number_of_subwords)

        guesses = (guess for guess in
                   self._softmax_iterator(joint_probabilities, target_word_length=len(contexts[0][1]))
                   if retry_wrong or guess not in previous_guesses)

        # Pad to top_n if there is < top_n guesses present
        top_n_guesses = list(islice(chain(guesses, repeat('_')), top_n))

        return top_n_guesses

    def _calculate_softmax_from_context(self, contexts: List[Tuple[str, str, str]],
                                        number_of_subwords: int) -> List[Tensor]:
        """
        Calculates the softmax tensors for a given list of contexts.
        :param contexts: Multiple contexts, where one context is a list of strings, with the mask word being MISSING.
        :param number_of_subwords: Number of subwords in the `missing_token` location
        :return: Softmax tensors for each context
        """

        softmax_tensors = []
        for left, _, right in contexts:
            unk_context = f'{left}  {" ".join(number_of_subwords * [self.tokenizer.mask_token])} {right}'
            bert_context = self.tokenizer(unk_context)['input_ids']
            softmax_tensor = \
                self._get_probabilities(self.tokenizer.decode(bert_context, clean_up_tokenization_spaces=True))
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

    def split_to_subwords(self, selected_word: str) -> List[int]:
        """
        Split the word to subwords returning word_ids

        :param selected_word:
        :return: the selected word, the word_ids and the frequency of the selected word
        """

        return self.tokenizer(selected_word, add_special_tokens=False)['input_ids']


def download(trie_pickle_fn='trie_words.pickle', wordlist_fn='resources/wordlist_3M.csv'):
    BertGuesser.prepare_resources(trie_pickle_fn, wordlist_fn)


if __name__ == '__main__':
    # just download and build everything if the module is not imported
    download()
