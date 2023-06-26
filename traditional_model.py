import argparse
import os

import pandas as pd 
import numpy as np
from pyvi import ViTokenizer
from sklearn import preprocessing 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm



def convert_to_onehot(labels):
    encoded_label = np.zeros((len(labels), len(set(labels))))
    
    for idx in range(len(labels)):
        encoded_label[idx][labels[idx]] = 1.0
        
    return encoded_label
        
        
def get_label_and_sentence(path, label_encoder):
    data = pd.read_csv(path)
    data.drop(columns=['Id'], inplace=True)
    # Convert string label to number
    data['Emotion'] = label_encoder.fit_transform(data["Emotion"])
    labels = data.iloc[:, 0].values
    sentences = data.iloc[:, 1].values
    
    return labels, sentences


def write_result(y_predict, y_test, sentences_test, save_folder):
    assert len(y_predict) == len(sentences_test)
    
    with open(os.path.join(save_folder, 'result.txt'), 'w') as f:
        for i in range(len(y_predict)):
            print("{}\t{}\t{}".format(y_predict[i], y_test[i], sentences_test[i]), file=f)
    
    
def load_data(data_folder, label_encoder):
    # Get label and sentence
    labels_train, sentences_train = get_label_and_sentence(os.path.join(data_folder,"train_nor_811.csv"), label_encoder=label_encoder)
    labels_test, sentences_test = get_label_and_sentence(os.path.join(data_folder,"test_nor_811.csv"), label_encoder=label_encoder)
    
    sentences_train = [ViTokenizer.tokenize(sentence) for sentence in sentences_train]
    sentences_test = [ViTokenizer.tokenize(sentence) for sentence in sentences_test]

    return (labels_train, sentences_train), (labels_test, sentences_test)

      
def main(args):
    # Encode label mapping text to number
    label_encoder = preprocessing.LabelEncoder()
    
    (labels_train, sentences_train), (labels_test, sentences_test) = load_data(
                                                                        data_folder=args.data_folder, 
                                                                        label_encoder=label_encoder
                                                                    )
    
    tf_vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
    
    x_train = tf_vectorizer.fit_transform(sentences_train)
    y_train = labels_train + 1

    x_test=tf_vectorizer.transform(sentences_test)
    y_test = labels_test + 1
    
    model = svm.LinearSVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    y_pred = label_encoder.inverse_transform(y_pred - 1)
    y_test = label_encoder.inverse_transform(y_test - 1)
    
    write_result(y_predict=y_pred, y_test=y_test, sentences_test=sentences_test, save_folder=args.save_folder)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="./dataset", required=True, type=str, help="Path to dataset folder")
    parser.add_argument("--save_folder", default="./result", type=str, help="Path to save result folder")
    

    args = parser.parse_args()

    main(args=args)