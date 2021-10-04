import unittest

import guessers


class BertGuesserTests(unittest.TestCase):
    bert_guesser = guessers.BertGuesser()

    def test_context_1(self):
        selected_word = 'fordul'
        contexts = [
            ['Nyugalmuk', 'sokszor', 'békés', 'alvásba', 'MISSING', ',', 'rengeteget', 'pihennek', ',', 'komótos']]
        number_of_subwords = 1
        expected_output = ['fordul', 'folyik', 'nyúlik', 'szakad', 'torkol', 'alakul', 'alszik', 'merült', 'átvált',
                           'szorul']
        previous_guesses = []
        model_output = self.bert_guesser.make_guess(contexts, len(selected_word), number_of_subwords,
                                                    previous_guesses=previous_guesses,
                                                    retry_wrong=False)
        self.assertEqual(model_output, expected_output)

    def test_context_2(self):
        selected_word = 'fordul'
        contexts = [
            ['Nyugalmuk', 'sokszor', 'békés', 'alvásba', 'MISSING', ',', 'rengeteget', 'pihennek', ',', 'komótos'],
            ['szélén', '–', 'ahol', 'rengeteg', 'kullancs', 'MISSING', 'elő', ',', 'akkor', 'lehet', ',']]

        number_of_subwords = 1
        expected_output = ['alakul', 'szalad', 'bukkan', 'nyúlik', 'szorul', 'léphet', 'lépett', 'robban', 'állhat',
                           'folyik']
        previous_guesses = {'került', 'fordul'}
        model_output = self.bert_guesser.make_guess(contexts, len(selected_word), number_of_subwords,
                                                    previous_guesses=previous_guesses,
                                                    retry_wrong=False)
        self.assertEqual(model_output, expected_output)

    def test_context_3(self):
        selected_word = 'Telefonos'
        contexts = [['Országos', 'MISSING', 'Ügyfélszolgálatunk', 'Kormányzati', 'Ügyfélvonallal', 'rendelkezik', 'és']]

        number_of_subwords = 2
        expected_output = ['Kormányon', 'Kormányos', 'Kormányok', 'Telefonon', 'Kormányát', 'Telefonos', 'Kormányra',
                           'Lakossági', 'Kormánymű', 'Telefonok']
        previous_guesses = {}
        model_output = self.bert_guesser.make_guess(contexts, len(selected_word), number_of_subwords,
                                                    previous_guesses=previous_guesses,
                                                    retry_wrong=False)
        self.assertEqual(model_output, expected_output)

    def test_context_4(self):
        selected_word = 'Telefonos'
        contexts = [['Országos', 'MISSING', 'Ügyfélszolgálatunk', 'Kormányzati', 'Ügyfélvonallal', 'rendelkezik', 'és'],
                    ['regisztrációtól', 'a', 'nyomtatványok', 'kitöltéséig', ';', 'MISSING', 'ügyintézés', ':',
                     'folyamatos', 'fejlesztéseink', 'eredményeképpen']]

        number_of_subwords = 2
        expected_output = ['Kormányok', 'Kormányon', 'Kormányos', 'hibaablak', 'Kormánymű', 'Telefonok', 'Telefonon',
                           'Ügyfélkör', 'Telefonos', 'Internetr']
        previous_guesses = {}
        model_output = self.bert_guesser.make_guess(contexts, len(selected_word), number_of_subwords,
                                                    previous_guesses=previous_guesses,
                                                    retry_wrong=False)
        self.assertEqual(model_output, expected_output)


class GensimTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
