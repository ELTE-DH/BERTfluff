from flask import Flask, request, current_app

from bertfluff.bert_guesser import BertGuesser
from bertfluff.gensim_guesser import GensimGuesser
from bertfluff.kenlm_guesser import KenLMGuesser


AVAILABLE_GUESSERS = {'bert': BertGuesser, 'cbow': GensimGuesser, 'kenlm': KenLMGuesser}


def parse_positive_ints(params, expected_params):
    """ Parse multiple positive integers from params with optional default value"""
    for name, default_value, error_message in expected_params:
        try:
            ret = int(params.get(name, default_value))
            if ret <= 0:
                raise ValueError
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
        - n_subwords: Number of subwords for the missing word.
        - prev_guesses: Previous guesses, represented as list.
        - retry_wrong: Bool. Possible values: y or n, true, false, yes, no, 1, 0
        - top_n: Number of guesses to return. Defaults to 10.
        - missing_token: Missing token. Defaults to `MISSING`.
    """

    # Choose
    guesser_name = params.get('guesser', '').lower()
    if guesser_name not in AVAILABLE_GUESSERS.keys():
        raise ValueError(f'guesser param must be one of {set(AVAILABLE_GUESSERS.keys())} !')

    # Positive ints
    try:
        number_of_subwords, top_n = \
            parse_positive_ints(params, (('no_of_subwords', 0, 'no_of_subwords must be positive int!'),
                                         ('top_n', 10, 'top_n must be positive int or must be omitted (default: 10)!')))
    except ValueError as e:
        raise e

    # Bool
    retry_wrong = str2bool(str(params.get('retry_wrong')), missing=None)
    if retry_wrong is None:
        raise ValueError('retry_wrong must be a bool represented'
                         ' one of the following values: y or n, true, false, yes, no, 1, 0 !')

    # Default: MISSING
    missing_token = params.get('missing_token')
    if missing_token is None:
        raise ValueError('missing_token must be a string which is an unique word in all contexts!')

    # List of strings can not be empty!
    raw_contexts = params.get('contexts[]', [])
    if len(raw_contexts) == 0:
        raise ValueError('contexts must be non-empty list of context strings!')

    # List of contexts must be splitted into (left, missing_word, right) triplets
    contexts = []   # List[Tuple[str, str, str]]
    for context in raw_contexts:
        context_words = context.split()
        ind = next((i for i, context_word in enumerate(context_words) if missing_token == context_word), None)
        if ind is None:
            raise ValueError('One elem of contexts does not contain missing_token!')
        contexts.append((' '.join(context_words[:ind]), context_words[ind], ' '.join(context_words[ind+1:])))

    # Default: empty list
    previous_guesses = params.get('prev_guesses[]', [])

    return guesser_name, contexts, number_of_subwords, previous_guesses, retry_wrong, top_n


def create_app():
    flask_app = Flask(__name__)
    flask_app.config['APP_SETTINGS'] = {'initialised_guessers': {k: v() for k, v in AVAILABLE_GUESSERS.items()}}

    @flask_app.route('/guess', methods=['GET', 'POST'])
    def guess():
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
            guesser_name, contexts, number_of_subwords, previous_guesses, retry_wrong,\
                top_n = parse_params(data)
        except ValueError as e:
            result = current_app.response_class(response=str(e), status=400, mimetype='application/json')
            return result

        selected_guesser = current_app.config['APP_SETTINGS']['initialised_guessers'][guesser_name]
        output = selected_guesser.make_guess(contexts, number_of_subwords, previous_guesses, retry_wrong,
                                             top_n)

        return {'guesses': output}

    @flask_app.route('/no_of_subwords', methods=['GET', 'POST'])
    def no_of_subwords():
        # Accept GET and POST as well
        if request.method == 'POST':
            data = request.json
        else:  # GET
            data = request.args

        if 'word' not in data:
            return current_app.response_class(response='word must be specified!', status=400,
                                              mimetype='application/json')

        guesser_name = data.get('guesser', '').lower()
        if guesser_name not in AVAILABLE_GUESSERS.keys():
            return current_app.response_class(response=f'guesser param must be one of'
                                                       f' {set(AVAILABLE_GUESSERS.keys())} !',
                                              status=400, mimetype='application/json')

        selected_guesser = current_app.config['APP_SETTINGS']['initialised_guessers'][guesser_name]
        output = len(selected_guesser.split_to_subwords(data['word']))

        return {'no_of_subwords': output}

    @flask_app.route('/word_similarity', methods=['GET', 'POST'])
    def word_similarity():
        # Accept GET and POST as well
        if request.method == 'POST':
            data = request.json
        else:  # GET
            data = request.args

        if 'word1' not in data or 'word2' not in data:
            return current_app.response_class(response='word1 and word2 must be specified!', status=400,
                                              mimetype='application/json')

        guesser_name = data.get('guesser', '').lower()
        if guesser_name not in AVAILABLE_GUESSERS.keys():
            return current_app.response_class(response=f'guesser param must be one of'
                                                       f' {set(AVAILABLE_GUESSERS.keys())} !',
                                              status=400, mimetype='application/json')

        selected_guesser = current_app.config['APP_SETTINGS']['initialised_guessers'][guesser_name]
        try:
            output = selected_guesser.word_similarity(data['word1'], data['word2'])
        except KeyError:
            output = -1

        return {'word_similarity': str(output)}

    return flask_app


if __name__ == '__main__':
    app = create_app()
    app.run()  # run our Flask app
