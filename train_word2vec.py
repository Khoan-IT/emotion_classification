import time
import argparse
import os

import pandas as pd

from gensim.models import Word2Vec
from pyvi import ViTokenizer


def get_label_and_sentence(path):
    data = pd.read_csv(path)
    data.drop(columns=['Id'], inplace=True)
    sentences = data.iloc[:, 1].values
    
    return sentences

def load_data(data_folder):
    # Get label and sentence
    sentences_train = get_label_and_sentence(os.path.join(data_folder,"train_nor_811.csv"))
    sentences_train = [ViTokenizer.tokenize(sentence) for sentence in sentences_train]
    return sentences_train

def train(data):
    start = time.time()
    w2v_model = Word2Vec(data, min_count = 1, vector_size = 300, window = 3, sg = 0)
    w2v_model.build_vocab(data,update = True)

    w2v_model.train(data, total_examples = w2v_model.corpus_count, epochs = 100)
    w2v_model.wv.save('vi-model-CBOW.bin')
    print('Training finish after: {} s!'.format(time.time() - start))


def main(args):
    data = load_data(args.data_folder)
    train(data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="./dataset", required=True, type=str, help="Path to dataset folder")
    # parser.add_argument("--save_folder", default="./result", type=str, help="Path to save result folder")
    

    args = parser.parse_args()

    main(args=args)