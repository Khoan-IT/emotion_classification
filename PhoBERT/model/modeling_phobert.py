import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier
from .losses import ContrastiveLoss, FocalLoss

class PhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(PhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        self.config = config
        self.contrastiveloss = ContrastiveLoss(1.5)
        self.focal_loss = FocalLoss(torch.FloatTensor([0.0, 0.01, 0.01, 0.98]).to('cuda'))
        self.count = 0
        self.do_eval = False

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        pooled_output = outputs[1]  # [CLS]
        intent_state = pooled_output

        intent_state = F.normalize(intent_state, p=2, dim=1)

        num_sentences = int(intent_state.shape[0] // 2)

        intent_logits = self.intent_classifier(intent_state)

        total_loss = 0
        # if intent_label_ids is not None:
        #     if self.num_intent_labels == 1:
        #         intent_loss_fct = nn.MSELoss()
        #         intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
        #     else:
        #         # intent_loss_fct = nn.CrossEntropyLoss()
        #         # intent_loss = intent_loss_fct(
        #         #     intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
        #         # )
        intent_loss_weight = 1
        if self.count != 0 and self.count % 1000 == 0:
            print("Decrease classifier weight!")
            intent_loss_weight -= 0.1

        if not self.do_eval:
            state_1, state_2 = torch.split(intent_state, num_sentences, dim=0)
            target_1, target_2 = torch.split(intent_label_ids, num_sentences, dim=0)
            new_target = torch.eq(target_1, target_2).long()

            intent_loss = self.focal_loss(intent_logits, intent_label_ids)
            total_loss += intent_loss_weight * intent_loss

            contrastive_loss = self.contrastiveloss(state_1, state_2, new_target)
            total_loss += (1 - intent_loss_weight) * contrastive_loss

        outputs = (intent_logits,) + outputs[2:]

        outputs = (total_loss,) + outputs

        self.count += 1

        return outputs
