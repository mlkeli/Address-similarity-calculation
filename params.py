import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset

EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 调用GPU

BERT_PATH = 'bert-base-chinese'  # 你自己的bert模型地址

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
MODEL_DIR = 'model/'  # 这是保存模型的地址，建在你代码的同一级即可