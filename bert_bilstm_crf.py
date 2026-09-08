import json
import os

import numpy as np
import torch
import torch.nn as nn

import modules
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader

import ner_dataset
import train_engine

# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = {'O': 0, 'B-dis': 1, 'I-dis': 2, 'B-sym': 3, 'I-sym': 4, 'B-mic': 5, 'I-mic': 6, 'B-ite': 7,
              'I-ite': 8, 'B-pro': 9, 'I-pro': 10, 'B-dep': 11, "I-dep": 12, 'B-dru': 13,
              'I-dru': 14, 'B-bod': 15, 'I-bod': 16, 'B-equ': 17, 'I-equ': 18,
              '[PAD]': 19, '[CLS]': 20, '[SEP]': 21, "[UNK]": 22}
verbose = False


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


# 训练模型
def train_model(model, **kwargs):
    train_engine.train(model, train_loader=kwargs['train_loader'], val_loader=kwargs['val_loader'],
                       train_dataset=kwargs['train_dataset'], epoch_num=kwargs['epoch'], is_crf=True,
                       model_save_pth=kwargs['model_save_pth'])


# 主函数
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def main():
    simple_train = True
    bert_model_name = './BertModel'  # 可使用其他 BERT 模型
    hidden_dim = 128
    tagset_size = len(label_dict)
    #model = modules.BERTBiLSTMCRF(bert_model_name, hidden_dim, tagset_size,verbose=False)
    model = modules.BERTBiLSTMCRF_Resnet_Attention(bert_model_name, hidden_dim, tagset_size)
    data = load_data()
    simple_data = data[:500]
    texts = []
    labels = []
    for dic in simple_data:
        texts.append(dic['text'])
        labels.append(dic['labels'])
    print("数据总量：", len(simple_data), len(texts), len(labels))

    # 划分训练集、验证集和测试集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)  # 80% 训练集，20% 临时集

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42)  # 将临时集平分为验证集和测试集

    print(f"训练集：{len(train_texts)}，验证集：{len(val_texts)}，测试集：{len(test_texts)}")

    # 创建数据集对象
    batch_size = 32
    epoch = 50


    if simple_train:
        batch_size = 8
        epoch = 5
    train_dataset = ner_dataset.NERDataset(train_texts, train_labels, label_dict, verbose=False)
    val_dataset = ner_dataset.NERDataset(val_texts, val_labels, label_dict, verbose=False)
    test_dataset = ner_dataset.NERDataset(test_texts, test_labels, label_dict, verbose=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 通常训练集要 shuffle
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取当前工作目录
    current_directory = os.getcwd()
    # 配置模型参数
    settings_dict = {'train_loader': train_loader, 'val_loader': val_loader, 'train_dataset': train_dataset,
                     'epoch': epoch,'model_save_pth':os.path.join(current_directory,"MNER_Bert_BiLSTM_CRF_Model")}
    print(f'Train Start With Epoch={epoch} Batch_Size={batch_size}')
    # 训练模型
    train_model(model, **settings_dict)


if __name__ == '__main__':
    main()
