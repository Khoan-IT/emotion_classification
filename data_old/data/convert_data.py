import pandas as pd
from statistics import mean

from pyvi import ViTokenizer

if __name__=="__main__":
    data = pd.read_csv("rawdata/review_electronic_devices_test.csv")
    new_data = data[["processed_content", "sentiment_3_cls"]]
    sentences = []
    labels = []
    lengths = []
    for idx, row in new_data.iterrows():
        content = str(row['processed_content'])
        content = ViTokenizer.tokenize(content.strip())
        lengths.append(len(content.split()))
        label = row['sentiment_3_cls']
        sentences.append(content)
        labels.append(label)

    with open("word-level/test/seq.in", 'w') as f:
        f.write("\n".join(sentences))
    
    with open("word-level/test/label", 'w') as f:
        f.write("\n".join(labels))