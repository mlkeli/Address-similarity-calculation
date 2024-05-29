import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score

from load_data import DataPrecessForSentence
from params import *
from model import MyModel

class Trainer():
    def __init__(self):
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        train_file = "/kaggle/input/ceshi05289963/train.txt"
        train_data = DataPrecessForSentence(bert_tokenizer, train_file)
        self.train_loader = DataLoader(train_data, batch_size=10, shuffle=False)

        train_file = "/kaggle/input/ceshi05289963/val.txt"
        dev_data = DataPrecessForSentence(bert_tokenizer, train_file)
        self.dev_loader = DataLoader(dev_data, batch_size=1, shuffle=False)

        self.model = MyModel(2).to(DEVICE)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    def train(self):
        tbar = tqdm(self.train_loader)
        self.model.train()
        train_loss = 0.0
        for batch_idx, data in enumerate(tbar):
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = data
            seqs, masks, segments, labels = batch_seqs.to(DEVICE), batch_seq_masks.to(DEVICE), batch_seq_segments.to(DEVICE), batch_labels.to(DEVICE)
            self.optimizer.zero_grad()
            loss, logits, probabilities = self.model(seqs, masks, segments, labels)
            loss.backward()
            self.scheduler.step()
            train_loss += loss.item()
            tbar.set_description("Loss: %.3f"%(train_loss / (batch_idx + 1)))

        # torch.save(model, MODEL_DIR + f'model_{epoch}.pth')

    def val(self):
        self.model.eval()
        all_pre = []
        all_tag = []
        tbar = tqdm(self.dev_loader)
        with torch.no_grad():
            for batch_idx, data in tqdm(tbar):
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = data
                seqs, masks, segments, labels = batch_seqs.to(DEVICE), batch_seq_masks.to(DEVICE), batch_seq_segments.to(DEVICE), batch_labels.to(DEVICE)
                self.optimizer.zero_grad()
                loss, logits, probabilities = self.model(seqs, masks, segments, labels)
                all_pre.extend(probabilities[:,1].cpu().numpy())
                all_tag.extend(batch_labels)
                val_loss += loss.item()
                tbar.set_description("Loss: %.3f"%(val_loss / (batch_idx + 1)))
        score = roc_auc_score(all_pre, all_tag)
        print(score)

        


if __name__ == "__main__":
    trainer = Trainer()

    for epoch in range(EPOCHS):
        trainer.train()
        trainer.val()