from flask import Flask, request

from guessers import BertGuesser, GensimGuesser

app = Flask(__name__)
available_guessers = {'bert': BertGuesser(), 'cbow': GensimGuesser()}


def parse_positive_ints(params, expected_params):
    """ Parse multiple positive integers from params with optional default value"""
    for name, default_value, error_message in expected_params:
        try:
            ret = int(params.get(name, default_value))
            if ret <= 0:
                raise ValueError()
        except ValueError as e:
            raise type(e)(error_message)

        yield ret


def str2bool(v, missing):
    """
    Original code from:
     https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    """
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        return missing


def parse_params(params):
    """
    Parse the parameters for the API. Will throw ValueError if data format is incorrect.
    The parameters are the following:
        - contexts: list of lists, where a context's words are separated by spaces, and contexts representad as list.
        - word_length: Number of characters of the missing word.
        - n_subwords: Number of subwords for the missing word.
        - prev_guesses: Previous guesses, represented as list.
        - retry_wrong: Bool. Possible values: y or n, true, false, yes, no, 1, 0
        - top_n: Number of guesses to return. Defaults to 10.
        - missing_token: Missing token. Defaults to `MISSING`.
    """

    # Choose
    guesser_name = params.get('guesser', '').lower()
    if guesser_name not in available_guessers.keys():
        raise ValueError(f'guesser param must be one of {set(available_guessers.keys())} !')

    # Positive ints
    try:
        word_length, number_of_subwords, top_n = \
            parse_positive_ints(params, (('word_len', None, 'word_len must be positive int!'),
                                         ('no_of_subwords', None, 'no_of_subwords must be positive int!'),
                                         ('top_n', 10, 'top_n must be positive int or must be omitted (default: 10)!')))
    except ValueError as e:
        raise e

    # Bool
    retry_wrong = str2bool(str(params.get('retry_wrong')), missing=None)
    if retry_wrong is None:
        raise ValueError('retry_wrong must be a bool represented'
                         ' one of the following values: y or n, true, false, yes, no, 1, 0 !')

    # List of strings can not be empty4
    contexts = [context.split() for context in params.get('contexts[]', [])]
    if len(contexts) == 0:
        raise ValueError('contexts must be list of context words !')

    # Default: MISSING
    missing_token = params.get('missing_token', 'MISSING')

    # Default: empty list
    previous_guesses = params.get('prev_guesses[]', [])

    return guesser_name, contexts, word_length, number_of_subwords, previous_guesses, retry_wrong, top_n, missing_token


@app.route('/guess', methods=['GET', 'POST'])
def bert_guess():
    # Accept GET and POST as well
    if request.method == 'POST':
        data = request.json
    else:  # GET
        # Fix Werkzeug behaviour:
        # https://werkzeug.palletsprojects.com/en/2.0.x/datastructures/#werkzeug.datastructures.MultiDict.to_dict
        data = {}
        for k, v in request.args.to_dict(flat=False).items():
            if k.endswith('[]'):
                data[k] = v
            else:
                data[k] = v[0]

    try:
        guesser_name, contexts, word_length, number_of_subwords, previous_guesses, retry_wrong, top_n, missing_token = \
            parse_params(data)
    except ValueError as e:
        result = app.response_class(response=str(e), status=400, mimetype='application/json')
        return result

    selected_guesser = available_guessers[guesser_name]
    output = selected_guesser.make_guess(contexts, word_length, number_of_subwords, previous_guesses, retry_wrong,
                                         top_n, missing_token)

    return {'guesses': output}


if __name__ == '__main__':
    app.run()  # run our Flask app
