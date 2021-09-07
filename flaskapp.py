from flask import Flask, request
import guessers

app = Flask(__name__)
try:
    bert_guesser = guessers.BertGuesser()
except Exception as e:
    print(e)
    bert_guesser = guessers.DummyGuesser()

try:
    gensim_guesser = guessers.GensimGuesser(model_fn='models/hu_wv.gensim')
except Exception as e:
    print(e)
    print('Gensim guesser is not running, creating a DummyGuesser.')
    gensim_guesser = guessers.DummyGuesser()


@app.route('/bertguess', methods=['GET', 'POST'])
def bert_guess():
    if request.method == 'POST':
        data = request.json
        output = bert_guesser.make_guess(contexts=data['contexts'], word_length=data['word_length'],
                                         number_of_subwords=data['number_of_subwords'],
                                         previous_guesses=data['previous_guesses'], retry_wrong=data['retry_wrong'],
                                         top_n=data['top_n'], missing_token=data['missing_token'])
        return {'guesses': output}
    else:
        return 'This is the BERT guesser\'s API.'


@app.route('/bertget/<contexts>/<word_length>/<n_subwords>/<prev_guesses>/<retry_wrong>/<top_n>/<missing_token>',
           methods=['GET'])
def bert_get(contexts, word_length, n_subwords, prev_guesses, retry_wrong, top_n, missing_token):
    """
    Get-based API. Will throw an exception if data format is incorrect.

    :param contexts: list of lists, where a context's words are separated by `_`, and contexts by `#`
    :param word_length: Number of characters of the missing word.
    :param n_subwords: Number of subwords for the missing word.
    :param prev_guesses: Previous guesses, separated by underscores.
    :param retry_wrong: y or n
    :param top_n: Number of guesses to return
    :param missing_token: Missing token. Defaults to `MISSING`.
    :return: a guess
    """
    contexts = [context.split('_') for context in contexts.split('#')]
    word_length = int(word_length)
    n_subwords = int(n_subwords)
    prev_guesses = prev_guesses.split('_')
    retry_wrong = retry_wrong == 'y'
    top_n = int(top_n)
    output = bert_guesser.make_guess(contexts=contexts, word_length=word_length,
                                     number_of_subwords=n_subwords,
                                     previous_guesses=prev_guesses, retry_wrong=retry_wrong,
                                     top_n=top_n, missing_token=missing_token)
    return {'guesses': output}


@app.route('/cbowguess', methods=['GET', 'POST'])
def cbow_guess():
    if request.method == 'POST':
        data = request.json
        output = gensim_guesser.make_guess(contexts=data['contexts'], word_length=data['word_length'],
                                           number_of_subwords=data['number_of_subwords'],
                                           previous_guesses=data['previous_guesses'], retry_wrong=data['retry_wrong'],
                                           top_n=data['top_n'], missing_token=data['missing_token'])
        return {'guesses': output}
    else:
        return 'This is the CBOW guesser\'s API.'



if __name__ == '__main__':
    app.run()  # run our Flask app
