import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import numpy as np
import ner_dataset

# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = {'O': 0, 'B-dis': 1, 'I-dis': 2, 'B-sym': 3, 'I-sym': 4, 'B-mic': 5, 'I-mic': 6, 'B-ite': 7,
              'I-ite': 8, 'B-pro': 9, 'I-pro': 10, 'B-dep': 11, "I-dep": 12, 'B-dru': 13,
              'I-dru': 14, 'B-bod': 15, 'I-bod': 16, '[PAD]': 17,'[CLS]': 18, '[SEP]': 19, "[UNK]": 20}

def load_data():
    data = []
    # 假设data为生成的文本和标签
    count = 0
    for i in open("data/专业化数据.txt", encoding='utf8'):
        line_dict = json.loads(i)
        if len(line_dict['labels']) == len(line_dict['text']):
            data.append(line_dict)
            count += 1
        else:
            print(len(line_dict['labels']), len(line_dict['text']), "自动剔除掉异常数据行")
    print("有效数据：", count, data[:5])
    return data


if __name__=='__main__':
    data = load_data()
    simple_data = data[:100]
    texts = []
    labels = []
    for dic in simple_data:
        texts.append(dic['text'])
        labels.append(dic['labels'])
    print("训练数据：", len(simple_data), len(texts), len(labels))

    batch_size = 16
    train_dataset = ner_dataset.NERDataset(texts, labels, label_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型并将其移动到GPU（如果可用）
    model = BertForTokenClassification.from_pretrained('BertModel', num_labels=len(label_dict)).to(device)

