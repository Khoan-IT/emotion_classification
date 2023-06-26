import argparse

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import preprocessing 


def read_result(path):
    with open(path, 'r') as f:
        lines = [line.split('\t')[:2] for line in f.readlines()]
    y_pred, y_test = list(zip(*lines))
    
    return y_pred, y_test


def visualize(y_pred, y_test, cls_label):
    ax = plt.subplot()
    df_cm = metrics.confusion_matrix(y_test, y_pred)
    sn.heatmap(df_cm, annot=True, ax=ax, fmt='g',cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(cls_label)
    ax.yaxis.set_ticklabels(cls_label)
    plt.show()
    

def report(y_pred, y_test, cls_label):
    print(metrics.classification_report(y_test, y_pred, target_names=cls_label))


def main(args):
    y_pred, y_test = read_result(args.path_result)
    
    label_encoder = preprocessing.LabelEncoder()
    
    y_pred = label_encoder.fit_transform(list(y_pred))
    
    y_test = label_encoder.transform(y_test)
    
    report(y_pred=y_pred, y_test=y_test, cls_label=label_encoder.classes_)
    
    visualize(y_pred=y_pred, y_test=y_test, cls_label=label_encoder.classes_)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--path_result", default="./result/svm/result.txt", type=str, help="Path to save result file")

    args = parser.parse_args()

    main(args=args)