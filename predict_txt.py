from tqdm import tqdm
from load_data import *
from model import *
from params import *
import os
import torch


def predict(text):
    train_filename = os.path.join(r'data/datatrain2', 'train.txt')
    train_text, train_label, max_length = read_data(train_filename)
    inputs = tokenizer.encode(text,return_tensors='pt')
    inputs = inputs.to(DEVICE)
    # 设置随机种子
    torch.manual_seed(0)
    model = torch.load(MODEL_DIR + '/datatrain2/model_19.pth', map_location=torch.device('cpu'))
    # 设置为评估模式
    model.eval()
    # 固定随机数生成器的状态
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        y_pre = model(inputs).reshape(-1)
        y_pre = torch.softmax(y_pre, dim=-1)
        label_idx = torch.argmax(y_pre)
    _, id2label = build_label_2_index(train_label)
    label = [id2label[label_idx.item()]]
    return label


if __name__ == '__main__':
    input_file = os.path.join(r'data', 'input.txt')
    output_file = os.path.join(r'data', 'output.txt')

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    for line in tqdm(lines, desc='Predicting', unit='line'):
        text = line.strip()
        result = predict(text)
        results.append(result[0])

    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(result + '\n')