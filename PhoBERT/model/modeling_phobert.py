import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier
from .losses import ContrastiveLoss, FocalLoss, GE2ELoss

class PhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(PhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert

        self.head_layer = nn.Linear(config.hidden_size, args.head_layer_dim) 

        self.intent_classifier = IntentClassifier(args.head_layer_dim, self.num_intent_labels, args.dropout_rate)

        self.config = config
        # self.focal_loss = FocalLoss(torch.FloatTensor([0.0, 0.01, 0.03, 0.96]).to(self.args.device))
        self.focal_loss = nn.CrossEntropyLoss()
        
        self.intent_loss_weight = 0.1

        if args.additional_loss == 'contrastiveloss':
            self.additional_loss = ContrastiveLoss(1.5)
        else:
            self.additional_loss = GE2ELoss()

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        if self.args.additional_loss == 'ge2eloss':
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            token_type_ids = token_type_ids.squeeze(0)
            # intent_label_ids = intent_label_ids.squeeze(0)

        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        pooled_output = outputs[1]  # [CLS]
        intent_state = pooled_output

        head_out = self.head_layer(intent_state)

        intent_logits = self.intent_classifier(head_out)
        # Get intent loss
        total_loss = 0
        if intent_label_ids is not None:
            # Get intent loss
            intent_loss = self.focal_loss(intent_logits, intent_label_ids)
            
            additional_loss = 0
            # Get addtional loss
            if self.args.additional_loss != 'None':
                if self.args.additional_loss == 'ge2eloss':
                    _, embedding_size = head_out.shape
                    head_out = head_out.reshape(self.num_intent_labels - 1, self.args.num_sample, embedding_size)

                    additional_loss = self.additional_loss(head_out)
                else:
                    head_out = F.normalize(head_out, p=2, dim=1)
                    num_sentences = int(head_out.shape[0] // 2)
                    state_1, state_2 = torch.split(head_out, num_sentences, dim=0)
                    target_1, target_2 = torch.split(intent_label_ids, num_sentences, dim=0)
                    new_target = torch.eq(target_1, target_2).long()
                    additional_loss = self.additional_loss(state_1, state_2, new_target)
                    
            # print(f"Intent Loss: {intent_loss}\n Additional loss: {additional_loss}")
                
            total_loss = self.intent_loss_weight * intent_loss + (1 - self.intent_loss_weight) * additional_loss

        outputs = (intent_logits,) + outputs[2:]

        outputs = (total_loss,) + outputs

        return outputs, head_out

