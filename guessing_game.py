import transformers
from collections import Counter
import csv
import random
import torch
import torch.nn.functional as F
from typing import Generator, List, Iterable


# suppress warning, only complain when halting
# transformers.logging.set_verbosity_error()


def create_corpora():
    """
    Used to create BERT-tokenized corpus. It is here for history reasons. It also deduplicates in a rudimentary manner.
    """
    c = Counter()
    sentences = set()
    dupes = 0
    tokenizer = transformers.AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', lowercase=True)
    with open('100k_corp.spl') as infile, open('tokenized_100k_corp.spl', 'w') as outfile:
        for line in infile:
            if line[0] == '#':
                continue
            tokens = tokenizer(line.strip(), add_special_tokens=False)
            sentence = tuple(tokens['input_ids'])
            if sentence not in sentences:
                sentences.add(sentence)
            else:
                dupes += 1
                continue

            for token in tokens['input_ids']:
                c[tokenizer.decode(token)] += 1
            outfile.write(tokenizer.decode(tokens['input_ids']))
            outfile.write('\n')

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

    @staticmethod
    def create_counter(filename: str, min_threshold: int = 20) -> Counter:
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

    def line_yielder(self, fname: str, word_id: int, full_sentence: bool) -> Generator:
        """
        With a word_id, it starts tokenizing the corpora (which is fast, hence not precomputed),
        and when finds a sentence containing the token, it yields the sentence.
        :param fname: corpus file in spl format
        :param word_id:
        :param full_sentence:
        :return:
        """
        with open(fname) as f:
            for line in f:
                tokens = self.tokenizer(line.strip(), add_special_tokens=False)
                if word_id in tokens['input_ids']:
                    if full_sentence:
                        yield line.strip()
                    else:
                        yield self.create_context(tokens['input_ids'], target_word=self.tokenizer.decode([word_id]))
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

    def create_context(self, sentence: List[int], target_word: str, window_size: int = 3) -> str:
        """
        In order to create a not subword-based context, we have to first reconstruct the original sentence,
        then find the word containing the subword, then rebuild and return the context.
        :param sentence: list of BERT token ids
        :param target_word: target word
        :param window_size: size of the window
        :return: a part of the original sentence containing the target word in the center
        """
        full_sentence = self.tokenizer.decode(sentence).split(' ')
        center = [(i, word) for i, word in enumerate(full_sentence) if target_word in word]
        assert len(center) >= 1, f'Center is {center}, should be at least one.'
        center = center[0][0]

        return ' '.join(full_sentence[max(0, center-window_size):min(len(full_sentence), center+window_size+1)])
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

    def guessing_game(self, show_bert_output: bool = True, full_sentence: bool = False) -> List:
        """
        Provides the interface for the game.
        :return: a list of length 3, containing the number of guesses of the player, BERT and the word missing
        """
        selected_word = random.choice(list(self.counter.keys()))
        selected_wordid = self.tokenizer.vocab[selected_word]
        guesses = set()
        user_guessed = False
        bert_guessed = False
        retval = [-1, -1, selected_word]
        softmax_tensors = []
        sentences = []

        print(len(selected_word), selected_wordid, self.counter[selected_word])

        for i, orig_sentence in enumerate(self.line_yielder('tokenized_100k_corp.spl', selected_wordid, full_sentence)):

            masked_sentence = orig_sentence.replace(selected_word, self.tokenizer.mask_token, 1)
            softmax_tensors.append(self.get_probabilities(masked_sentence))
            bert_guesses = self.make_guess(softmax_tensors, word_length=len(selected_word),
                                           previous_guesses=guesses, retry_wrong=False)

            # UI
            current_sentence = orig_sentence.replace(selected_word, '#'*len(selected_word), 1)
            sentences.append(current_sentence)
            print('\n'.join(create_aligned_text(sentences)))
            print('-'*80)

            if not user_guessed:
                user_input = input('Please input your guess: ')
                if user_input.strip() == selected_word:
                    user_guessed = True
                    retval[0] = i+1

            print(f'BERT\'s guess is {bert_guesses[0]}')

            if show_bert_output:
                print(f'BERT\'s top 5 guesses: {" ".join(bert_guesses[:5])}')
            guess = bert_guesses[0]

            if not bert_guessed:
                if guess == selected_word:
                    print('BERT guessed the word.')
                    bert_guessed = True
                    retval[1] = i+1
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
