import requests.utils

from utils.misc import bert_get_convert

if __name__ == '__main__':
    top_n = 10
    missing_token = 'MISSING'
    retry_wrong = False

    word_length = 9
    number_of_subwords = 1
    previous_guesses = []

    context_1 = [
        ['b)', 'az', 'adatállományok', 'helyreállításának', 'lehetőségét', 'MISSING', 'intézkedésekről', ',', 'ezen',
         'belül', 'a']]
    output_1 = requests.post('http://127.0.0.1:5000/bertguess',
                             json={'contexts': context_1, 'word_length': word_length,
                                   'number_of_subwords': number_of_subwords,
                                   'previous_guesses': previous_guesses,
                                   'retry_wrong': retry_wrong, 'top_n': top_n, 'missing_token': missing_token})

    print(output_1)
    print(output_1.json())

    output_2 = requests.post('http://127.0.0.1:5000/cbowguess',
                             json={'contexts': context_1, 'word_length': word_length,
                                   'number_of_subwords': number_of_subwords,
                                   'previous_guesses': previous_guesses,
                                   'retry_wrong': retry_wrong, 'top_n': top_n, 'missing_token': missing_token})

    print(output_2)
    print(output_2.json())

    url_extension_1 = bert_get_convert(context_1, word_length, number_of_subwords, previous_guesses, retry_wrong, top_n,
                                       missing_token)

    print(url_extension_1)

    output_3 = requests.get(f'http://127.0.0.1:5000/bertget/{url_extension_1}')

    print(output_3)
    print(output_3.json())

    context_2 = [
        ['Nyugalmuk', 'sokszor', 'békés', 'alvásba', 'MISSING', ',', 'rengeteget', 'pihennek', ',', 'komótos'],
        ['szélén', '–', 'ahol', 'rengeteg', 'kullancs', 'MISSING', 'elő', ',', 'akkor', 'lehet', ',']]

    previous_guesses_2 = {'került', 'fordul'}

    url_extension_2 = bert_get_convert(context_2, word_length, number_of_subwords, previous_guesses_2, retry_wrong,
                                       top_n, missing_token)

    output_4 = requests.get(f'http://127.0.0.1:5000/bertget/{url_extension_2}')

    print(output_4)
    print(output_4.json())
