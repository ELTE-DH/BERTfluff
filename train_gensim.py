import logging
from os. path import join as os_path_join

import gensim


def main(input_corpus, resources_dir: str = 'resources', models_dir='models'):
    logging.basicConfig(filename='gensim.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.DEBUG)
    sentences = gensim.models.word2vec.LineSentence(os_path_join(resources_dir, input_corpus))
    model = gensim.models.Word2Vec(max_vocab_size=1_000_000, workers=8, window=5, vector_size=100, sg=0)
    model.build_vocab(sentences)
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    model.save(os_path_join(models_dir, 'hu_wv.gensim'))


if __name__ == '__main__':
    main('corp_10M.spl', '/home/levaid/Downloads/')  # TODO
