import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from load_data import DataPrecessForSentence
from params import *
from model import BertModel

class Trainer():
    def __init__(self, model_path):
        # 加载预训练模型的权重并移至 CPU
        self.model = BertModel()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'],strict=False)
        self.model.eval()

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        dev_file = "data/val.txt"
        dev_data = DataPrecessForSentence(bert_tokenizer, dev_file)
        self.dev_loader = DataLoader(dev_data, batch_size=1, shuffle=False)

    def test(self):
        tbar = tqdm(self.dev_loader)
        with torch.no_grad():S
            for batch_idx, data in tqdm(enumerate(tbar)):
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = data
                seqs, masks, segments, labels = batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels
                loss, logits, probabilities = self.model(seqs, masks, segments, labels)
                pre = torch.argmax(probabilities,dim=1)
                print(pre)

if __name__ == "__main__":
    model_path = "model/best.pth.tar"  # 替换为你的.pth.tar模型文件路径
    trainer = Trainer(model_path)

    trainer.test()
