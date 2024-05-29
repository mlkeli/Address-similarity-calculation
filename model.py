import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels = 2)  # /bert_pretrain/
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        output = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)
        loss = output['loss']
        logits = output['logits']
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities