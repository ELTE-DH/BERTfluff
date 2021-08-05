import requests
import json

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