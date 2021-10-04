from copy import deepcopy
from urllib.parse import urlencode

import requests

SERVER = 'http://127.0.0.1:5000'


def main():

    params = {
              # 'guesser': guesser_type,
              # 'contexts[]': [' '.join(context) for context in context_1],
              'word_len': 9,
              'no_of_subwords': 1,
              'prev_guesses[]': [],
              'retry_wrong': False,
              'top_n': 10,
              'missing_token': 'MISSING'
             }

    ####################################################################################################################

    context_1 = [
        ['b)', 'az', 'adatállományok', 'helyreállításának', 'lehetőségét', 'MISSING', 'intézkedésekről', ',', 'ezen',
         'belül', 'a']]

    params_1 = deepcopy(params)
    params_1['contexts[]'] = [' '.join(context) for context in context_1]
    for guesser_type in ('bert', 'cbow'):
        params_1['guesser'] = guesser_type
        output_1 = requests.post(f'{SERVER}/guess', json=params_1)

        print('POST', guesser_type, output_1)
        print(output_1.json())

    for guesser_type in ('bert', 'cbow'):
        params_1['guesser'] = guesser_type
        output_2 = requests.get(f'{SERVER}/guess?{urlencode(params_1, doseq=True)}')

        print('GET', guesser_type, output_2)
        print(output_2.json())

    ####################################################################################################################

    context_2 = [
        ['Nyugalmuk', 'sokszor', 'békés', 'alvásba', 'MISSING', ',', 'rengeteget', 'pihennek', ',', 'komótos'],
        ['szélén', '–', 'ahol', 'rengeteg', 'kullancs', 'MISSING', 'elő', ',', 'akkor', 'lehet', ',']]

    previous_guesses_2 = ['került', 'fordul']

    params_2 = deepcopy(params)
    params_2['contexts[]'] = [' '.join(context) for context in context_2]
    params_2['prev_guesses[]'] = previous_guesses_2

    for guesser_type in ('bert', 'cbow'):
        params_2['guesser'] = guesser_type
        output_3 = requests.post(f'{SERVER}/guess', json=params_2)

        print('POST', guesser_type, output_3)
        print(output_3.json())

    for guesser_type in ('bert', 'cbow'):
        params_2['guesser'] = guesser_type
        output_4 = requests.get(f'{SERVER}/guess?{urlencode(params_2, doseq=True)}')

        print('GET', guesser_type, output_4)
        print(output_4.json())


if __name__ == '__main__':
    main()
