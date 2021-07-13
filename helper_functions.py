import transformers
from collections import Counter
import csv
import random
import torch
import torch.nn.functional as F

TOKENIZER = transformers.AutoTokenizer.from_pretrained(
    'SZTAKI-HLT/hubert-base-cc', lowercase=True)
model = transformers.BertForMaskedLM.from_pretrained(
    'SZTAKI-HLT/hubert-base-cc', return_dict=True)


def create_corpora():

    c = Counter()
    with open('100k_corp.spl') as infile, open('tokenized_100k_corp.spl', 'w') as outfile:
        for line in infile:
            if line[0] == '#':
                continue
            tokens = TOKENIZER(line.strip(), add_special_tokens=False)
            for token in tokens['input_ids']:
                c[TOKENIZER.decode(token)] += 1
            outfile.write(TOKENIZER.decode(tokens['input_ids']))
            outfile.write('\n')

    with open('freqs.csv', 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for word, freq in sorted(c.items(), key=lambda x: x[1], reverse=True):
            csv_writer.writerow([word, freq])


# get_sentences
c = Counter()
min_threshold = 20
with open('freqs.csv') as infile:
    csv_reader = csv.reader(infile)
    for word, freq in csv_reader:
        if int(freq) < min_threshold or not word.isalpha() or len(word) <= 5:
            continue
        else:
            c[word] = int(freq)


def line_yielder(fname, word_id):
    with open(fname) as f:
        for line in f:
            tokens = TOKENIZER(line.strip())
            if word_id in tokens['input_ids']:
                yield line.strip()
            else:
                continue


def topn_fixed_length(mask_word, top_n, word_length, tokenizer, check_word_length=True):
    order = torch.argsort(mask_word, dim=1, descending=True)[0]
    possible_words = []
    for token_id in order:
        token = tokenizer.decode([token_id])
        if check_word_length:
            if len(token) == word_length:
                possible_words.append(token)
        else:
            possible_words.append(token)
        if len(possible_words) == top_n:
            break
    return possible_words


def guessgame() -> int:
    selected_word = random.choice(list(c.keys()))
    selected_wordid = TOKENIZER.vocab[selected_word]
    print(len(selected_word), selected_wordid, c[selected_word])
    guesses = set()
    user_guessed = False
    bert_guessed = False
    retval = [-1, -1, selected_word, '', '']

    for i, orig_sentence in enumerate(line_yielder('tokenized_100k_corp.spl', selected_wordid)):

        masked_sentence = orig_sentence.replace(
            selected_word, TOKENIZER.mask_token, 1)
        print(orig_sentence.replace(selected_word, '#'*len(selected_word)))
        tokenized_text = TOKENIZER(masked_sentence, return_tensors='pt')
        mask_index = torch.where(
            tokenized_text["input_ids"][0] == TOKENIZER.mask_token_id)
        output = model(**tokenized_text)
        logits = output.logits
        softmax = F.softmax(logits, dim=-1)
        mask_word = softmax[0, mask_index, :]

        top_n = topn_fixed_length(mask_word, 20, word_length=len(
            selected_word), tokenizer=TOKENIZER, check_word_length=True)
        remaining_words = [word for word in top_n if word not in guesses]
        if not user_guessed:
            user_input = input(prompt='Please input your guess: ')
            if user_input.strip() == selected_word:
                user_guessed = True
                retval[0] = i+1

        print(
            f'BERT\'s top 5 guesses (out of which the 1st one is the guess): {" ".join(remaining_words[:5])}')
        guess = remaining_words[0]

        if not bert_guessed:
            if guess == selected_word:
                print('BERT guessed the word.')
                bert_guessed = True
                retval[1] = i+1
            else:
                guesses.add(guess)

        if bert_guessed and user_guessed:
            return retval


game_lengths = [guessgame() for i in range(5)]

