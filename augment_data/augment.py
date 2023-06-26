import random
import io
import sys
import argparse
import os
import csv
import torch
import re

import numpy as np
import pandas as pd

from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options
from pyvi import ViTokenizer
from tqdm import tqdm
from string import punctuation
from copy import deepcopy



def load_vectors(fname, num_words = 10000):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = fin.readline().split()
    print('Mô hình có',n,'từ vựng')
    print('Mỗi từ được biểu diễn bằng một vector',d,'chiều')
    print('Đang tải...')
    data = {}
    for idx, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        # get embedding vector
        data[tokens[0]] = np.array([float(val) for val in tokens[1:]])
        # normalize embedding vector
        data[tokens[0]] /= np.linalg.norm(data[tokens[0]])
        if idx >= num_words:
            break
    print('Đã tải {} vector tiếng Việt'.format(num_words))
    return data


def find_similar_word(word, word2vec, num_similar = 1):
    ls_similar_word = []
    if word not in word2vec:
        return []
    ref_v = word2vec[word]

    top_num = 20
    top_w = ['']*top_num
    top_s = [-1]*top_num

    for k in word2vec.keys():
        if word == k:
            continue
        score =  np.dot(ref_v, word2vec[k])
        if score < np.min(top_s):
            continue
        for i in range(top_num):
            if score >= top_s[i]:
                top_w[i+1:] = top_w[i:top_num-1]
                top_w[i] = k
                top_s[i+1:] = top_s[i:top_num-1]
                top_s[i] = score
                break

    count = 0
    for i in range(top_num):
        if count >= num_similar:
          return ls_similar_word
        if top_w[i].lower() == word:
          continue
        ls_similar_word.append(top_w[i])
        count += 1
    return []


def get_phobert_model():
    phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phoBERT.to(device)
    phoBERT.eval()

    # Initialize Byte Pair Encoding for PhoBERT
    class BPE():
        bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'
    args = BPE()
    phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
    
    return phoBERT

     
def data_aug(sentence, phoBERT, word2vec, num_aug=1):
    ls_new_sentence = []
    token = ViTokenizer.tokenize(sentence).split()
    ls_synonym = []
    for idx in range(len(token)):
        # relative words
        if token[idx] in ['thì', 'cũng', 'nhưng', 'không', 'được', 'nếu', 'thôi']:
            continue

        ls_similar_word = [_.lower() for _ in find_similar_word(token[idx], word2vec, 10)]
        tmp_mask = token.copy()
        tmp_mask[idx] = '<mask>'
        ls_roberta_fill_in = [_[2].lower() for _ in phoBERT.fill_mask(' '.join(tmp_mask), topk=50)]
        word = ''
        for similar_word in ls_similar_word:
            if similar_word in ls_roberta_fill_in:
                word = similar_word
                break
        if word != '':
            ls_synonym.append(word)
        else:
            ls_synonym.append(token[idx])
    
    for _ in range(num_aug):
        if len(ls_synonym) >= 4:
            idx_word_replace = random.choices(range(len(ls_synonym)), k=int(0.5*len(ls_synonym)))
            tmp = token.copy()
            for index in idx_word_replace:
                tmp[index] = ls_synonym[index]
            ls_new_sentence.append(' '.join(tmp))
    
    return list(set(ls_new_sentence))


def write_result(result, path_save):
    header = ["Id", "Emotion", "Sentence"]
    with open(path_save, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for idx, line in enumerate(result):
            writer.writerow([idx, *line])


def read_data(path):
    data = pd.read_csv(path)
    data.drop(columns=['Id'], inplace=True)
    
    return data


def remove_accent(s):
    s = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', s)
    s = re.sub('[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'A', s)
    s = re.sub('[éèẻẽẹêếềểễệ]', 'e', s)
    s = re.sub('[ÉÈẺẼẸÊẾỀỂỄỆ]', 'E', s)
    s = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', s)
    s = re.sub('[ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]', 'O', s)
    s = re.sub('[íìỉĩị]', 'i', s)
    s = re.sub('[ÍÌỈĨỊ]', 'I', s)
    s = re.sub('[úùủũụưứừửữự]', 'u', s)
    s = re.sub('[ÚÙỦŨỤƯỨỪỬỮỰ]', 'U', s)
    s = re.sub('[ýỳỷỹỵ]', 'y', s)
    s = re.sub('[ÝỲỶỸỴ]', 'Y', s)
    s = re.sub('đ', 'd', s)
    s = re.sub('Đ', 'D', s)
    return s


def main(args):
    none_augment = ["Enjoyment", "Disgust", "Sadness", "Other"]
    
    word2vec = load_vectors('./fasttext_vn/cc.vi.300.vec', num_words=args.num_word)
    phobert = get_phobert_model()
    
    data = read_data(os.path.join(args.data_folder, args.name_data))
    result = []
    
    for i, sentence in tqdm(enumerate(data['Sentence'])):
        result.append([data['Emotion'][i], sentence])
        if data['Emotion'][i] in none_augment:
            continue
        
        # Generate synonyms
        
        new_sentences = data_aug(sentence=sentence, phoBERT=phobert, word2vec=word2vec, num_aug=args.num_aug)
        if len(new_sentences) != 0:
            for ns in new_sentences:
                result.append([data['Emotion'][i], ns])
                
        # Remove accent
        
        # sentence = sentence.replace("_", "")
        # tokens = sentence.split()
        # percent_remove = random.random()
        # idx_remove_accent = random.choices(range(len(tokens)), k=int(percent_remove*len(tokens)))
        # tmp_tokens = deepcopy(tokens)
        # for idx in idx_remove_accent:
        #     tmp_tokens[idx] = remove_accent(tokens[idx])
        # result.append([data['Emotion'][i], " ".join(tmp_tokens)])
        
    
    write_result(result, os.path.join(args.save_folder, args.name_data))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_word", default=10000, required=True, type=int, help="Number of words to get similar words")
    parser.add_argument("--data_folder", default="./dataset/original", required=True, type=str, help="Path to folder consisting original data")
    parser.add_argument("--save_folder", default="./dataset/augment_phobert", required=True, type=str, help="Path to folder save data after augmentation")
    parser.add_argument("--name_data", default="train.csv", required=True, type=str, help="Name of dataset to train")
    parser.add_argument("--num_aug", default=1, required=True, type=int, help="Number of times to generate data")
    args = parser.parse_args()

    main(args=args)
    
    
