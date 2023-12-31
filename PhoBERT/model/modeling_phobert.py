import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier

class PhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(PhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.list_cnn = nn.ModuleList([nn.Conv1d(config.hidden_size, 256, i, device='cuda', padding='same') for i in range(3,6)])
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)

        self.config = config

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        hidden_state = torch.transpose(outputs.last_hidden_state[:,1:-1,:], 1, 2)
        cnn_outputs = []
        for layer in self.list_cnn:
            output = layer(hidden_state)
            output, _ = torch.max(output, dim=-1)
            cnn_outputs.append(output)

        intent_state = torch.cat(cnn_outputs, dim=1)
        # hidden_state = outputs.last_hidden_state[:,1:-1,:]
        # _, (h_n, c_n) = self.lstm(hidden_state)
        
        # state = h_n
        # intent_state = torch.squeeze(state)
        
        # pooled_output = outputs[1]  # [CLS]
        # intent_state = pooled_output

        intent_logits = self.intent_classifier(intent_state)

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
            # total_loss += intent_loss


        outputs = (intent_logits,) + outputs[2:]

        outputs = (total_loss,) + outputs

        return outputs
