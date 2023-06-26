#!/usr/bin/python -tt
# -*- coding: utf-8
import openai
import os
import json
import pickle
import re
import random
import time

from tqdm import tqdm
import pandas as pd
from collections import defaultdict

key = ''
# openai.api_key = ''  # Set your API key as an environment variable

MODEL = "gpt-3.5-turbo"  # Choose the GPT model you want to use

def process_chatgpt_result(gpt_result):
    sents = gpt_result.split('\n')
    results = []
    for s in sents:
        results.append(" ".join(s.split()[1:]).replace('.', '').lower())
    return results
 
def generate_synonym(prompt):

    openai.api_key = key
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=prompt,
        )
    completion_text = response.choices[0].message.content
    
    return process_chatgpt_result(completion_text)

def get_traveled_id(path_save):
    if os.path.exists(path_save):
        data = json.load(open(path_save, 'r'))
        if len(data) != 0:
            return list(data.keys())
    
    return []

if __name__ == "__main__":
    none_augment = ["Enjoyment", "Disgust", "Sadness", "Other"]
    path_save = "./result/new_data.json"
    
    prompt = [
                {"role": "user", "content": ""},
            ]
    results = {}
    traveled_id = get_traveled_id(path_save)
    data = pd.read_csv("./dataset/train_nor_811.csv")

    for times, sentence in tqdm(enumerate(data["Sentence"])):
        if data["Emotion"][times] in none_augment or str(times) in traveled_id:
            continue
        
        sentence = sentence.lower()
        question = "sinh cho tôi 3 câu tương tự câu này: {}".format(sentence)
        prompt[0]['content'] = question
        new_sentence = []
        try:
            new_sentence += generate_synonym(prompt)
            time.sleep(10)
        except:
            print('Bad Gateway, sleeping for 60s')
            time.sleep(60)
            new_sentence += generate_synonym(prompt)
        new_sentence = list(set(new_sentence))
        traveled_id.append(data["Id"][times])
        
        if times % 5 == 0:
            results[times] = new_sentence
            save_file = open(os.path.join(path_save), 'w', encoding='utf-8')
            json.dump(results, save_file, ensure_ascii=False)
            save_file.close()
            
        results[times] = new_sentence
        save_file = open(os.path.join(path_save), 'w', encoding='utf-8')
        json.dump(results, save_file, ensure_ascii=False)
        save_file.close()
