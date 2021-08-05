from flask import Flask, request
from bert_guesser import Bert_Guesser
from gensim_guesser import GensimGuesser

app = Flask(__name__)
bert_guesser = Bert_Guesser(trie_fn='models/trie_words.pickle')
gensim_guesser = GensimGuesser(model_fn='models/hu_wv.gensim')


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
