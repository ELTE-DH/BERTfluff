import gensim


class GensimHelper:
    def __init__(self, model_fn='models/hu_wv.gensim'):
        self.model = gensim.models.Word2Vec.load(model_fn)

    def word_similarity(self, word_1: str, word_2: str) -> float:
        """
        Calculates similarity by taking the cosine similarity of the vectors of `word_1` and `word_2`.
        Returns -1 if word is not in the vocabulary.

        :param word_1: One word
        :param word_2: The other word
        :return: Similarity of the corresponding word vectors.
        """
        for word in [word_1, word_2]:
            if word not in self.model.wv:
                print(f'{word} not in vocabulary!')
                return -1.0

        return self.model.wv.similarity(word_1, word_2)
