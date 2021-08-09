import gensim
import logging


def main():
    logging.basicConfig(filename='gensim.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.DEBUG)
    sentences = gensim.models.word2vec.LineSentence('/home/levaid/Downloads/corp_10M.spl')
    model = gensim.models.Word2Vec(max_vocab_size=1_000_000, workers=8, window=5, vector_size=100, sg=0)
    model.build_vocab(sentences)
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    model.save('models/hu_wv.gensim')


if __name__ == '__main__':
    main()
