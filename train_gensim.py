import gensim
import gzip
import logging

logging.basicConfig(filename='gensim.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.DEBUG)


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, corpus_path: str):

        self.corpus_path = corpus_path

    def __iter__(self):

        if self.corpus_path[-3:] == '.gz':
            with gzip.open(self.corpus_path, 'rt') as infile:
                for line in infile:
                    yield gensim.utils.simple_preprocess(line, min_len=2, max_len=15)
        else:
            with open(self.corpus_path, 'r') as infile:
                for line in infile:
                    yield gensim.utils.simple_preprocess(line, min_len=2, max_len=15)


if __name__ == '__main__':
    # sentences = MyCorpus('/home/levaid/Downloads/corp_10M.spl')
    sentences = gensim.models.word2vec.LineSentence('/home/levaid/Downloads/corp_10M.spl')
    model = gensim.models.Word2Vec(max_vocab_size=1_000_000, workers=8, window=5, vector_size=100, sg=0)
    model.build_vocab(sentences)
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    model.save('models/hu_wv.gensim')
