import torch
import torch.nn as nn
import numpy as np

from .module import IntentClassifier
from gensim.models.keyedvectors import KeyedVectors



class ModelCNN(nn.Module):
    def __init__(self, args, tokenizer, intent_label_lst):
        super().__init__()
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self._initialize_embedding(tokenizer)
        
        self.list_cnn = nn.ModuleList([nn.Conv1d(args.embedding_dim, 256, i, device='cuda', padding='same') for i in range(3,6)])
        self.intent_classifier = IntentClassifier(args.hidden_size, self.num_intent_labels, args.dropout_rate)

    
    def _initialize_embedding(self, tokenizer):
        word_vectors = KeyedVectors.load("./cbow_model/vi-model-CBOW.bin")
        
        word_index = tokenizer.word_index
        vocabulary_size = min(len(word_index) + 1, self.args.max_vocab_size)
        embedding_matrix = np.zeros((vocabulary_size, self.args.embedding_dim))

        for word, i in word_index.items():
            if i >= self.args.max_vocab_size:
                continue
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25), self.args.embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
    
    
    def forward(self, input_ids, intent_label_ids):
        embedding = self.embedding(input_ids)
        embedding = torch.transpose(embedding, 1, 2)
        cnn_outputs = []
        for layer in self.list_cnn:
            output = layer(embedding)
            output, _ = torch.max(output, dim=-1)
            cnn_outputs.append(output)

        state = torch.cat(cnn_outputs, dim=1)

        intent_logits = self.intent_classifier(state)

        total_loss = 0

        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += intent_loss


        outputs = (intent_logits, state)

        outputs = (total_loss,) + outputs

        return outputs 
