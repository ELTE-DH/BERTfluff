from flask import Flask, request
from bert_guesser import Bert_Guesser

app = Flask(__name__)
guesser = Bert_Guesser(trie_fn='models/trie_words.pickle')


@app.route('/bertguess', methods=['GET', 'POST'])
def bert_guess():
    if request.method == 'POST':
        data = request.json
        output = guesser.make_guess(contexts=data['contexts'], word_length=data['word_length'],
                                    number_of_subwords=data['number_of_subwords'],
                                    previous_guesses=data['previous_guesses'], retry_wrong=data['retry_wrong'],
                                    top_n=data['top_n'], missing_token=data['missing_token'])
        return {'guesses': output}
    else:
        return 'This is the BERT guesser\'s API.'


if __name__ == '__main__':
    app.run()  # run our Flask app
