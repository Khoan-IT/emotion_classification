import torch.utils.data
import argparse

import os
import yaml
import torch
import json

import numpy as np

from typing import List, Dict, Union
from transformers import (
    AutoTokenizer,
    RobertaConfig
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict 



CONFIG = {
    "pad_token_label_id": -100,
    "cls_token_segment_id": 0,
    "pad_token_segment_id": 0,
    "sequence_a_segment_id": 0,
    "mask_padding_with_zero": True,
}
    


def convert_input_to_tensor(words: List[str], label, tokenizer, args) -> Dict[str, torch.Tensor]:
    input_ids = []
    attention_mask = []
    token_type_ids = []

    tokens = []
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens

    # Account for [CLS] and [SEP]
    special_tokens_count = 2
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[: (args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [tokenizer.sep_token]
    token_type_ids = [CONFIG['sequence_a_segment_id']] * len(tokens)

    # Add [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_type_ids = [CONFIG['cls_token_segment_id']] + token_type_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1 if CONFIG['mask_padding_with_zero'] else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0 if CONFIG['mask_padding_with_zero'] else 1] * padding_length)
    token_type_ids = token_type_ids + ([CONFIG['pad_token_segment_id']] * padding_length)

    # Change to Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    intent_label_id = torch.tensor(label, dtype=torch.long)

    input_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'intent_label_ids': intent_label_id,
    }

    return input_data


def get_data(data_dir, label_file, args, tokenizer):
    with open(label_file, 'r') as f:
        classes = [l.strip() for l in f.readlines()]

    with open(os.path.join(data_dir, 'seq.in'), 'r') as f:
        sentences = [l.strip() for l in f.readlines()]
    
    with open(os.path.join(data_dir, 'label'), 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    assert len(sentences) == len(labels)
    data = []
    for sentence, label in zip(sentences, labels):
        sentence = sentence.strip().split()
        intent_label_id = int(classes.index(label))
        input_data = convert_input_to_tensor(sentence, intent_label_id, tokenizer, args)
        data.append(input_data)

    return data, labels

    

class SentenceLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_file, args, tokenizer):
        self.data, _ = get_data(data_dir, label_file, args, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def split_label(labels):
    name_labels = set(labels)
    dict_labels = {}
    labels = np.array(labels)
    for label in name_labels:
        dict_labels[f'{label}'] = np.where(labels == label)[0]
    return dict_labels


class GE2ESentenceLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_file, args, tokenizer):
        self.data, self.labels = get_data(data_dir, label_file, args, tokenizer)
        self.dict_labels = split_label(self.labels)
        self.num_sample = args.num_sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = {}
        for _, v in self.dict_labels.items():
            idxes = np.random.choice(v, self.num_sample, replace=False)
            for idx in idxes:
                for k, vm in self.data[idx].items():
                    if k not in result:
                        result[k] = vm.unsqueeze(0)
                    else:
                        result[k] = torch.concat((result[k], vm.unsqueeze(0)), dim=0)
        
        return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    data_dir = '../data/word-level/train'
    label_file = '../data/word-level/intent_label.txt'

    train_data = GE2ESentenceLoader(data_dir, label_file, args, tokenizer)

    data_loader = DataLoader(train_data, batch_size=2)

    for batch in data_loader:
        print(batch['intent_label_ids'].shape)
        exit()
