import json
from tqdm import tqdm
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
import torch
from torch import nn
import numpy as np

import pandas as pd
import numpy as np
from transformers import BertTokenizerFast

import bert_model

# 定义标签映射
label2id = {
    "O": 0,
    "B-ite": 1,
    "I-ite": 2,
    "B-dis": 3,
    "I-dis": 4,
    "B-bod": 5,
    "I-bod": 6,
    "B-sym": 7,
    "I-sym": 8,
    "B-pro": 9,
    "I-pro": 10,
    "[CLS]": 11,
    "[SEP]": 12,
    "[UNK]": 13
}

# 创建分词器
tokenizer = BertTokenizerFast.from_pretrained("BertModel")
# 设置最大序列长度
max_seq_len = 256
# 填充ID
pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
# 未知单词ID
unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
unword_label_id = len(label2id)


# 将数据转换为适合Bert模型的格式
def trans2TheDataset(data, max_seq_len):
    data2 = []
    for k, v in data:
        k = [tokenizer.convert_tokens_to_ids(i) for i in k]
        v = [label2id[i] for i in v]
        input_ids = [cls_id] + k
        labels = [unword_label_id] + v
        if len(input_ids) > max_seq_len - 1:
            input_ids = input_ids[:max_seq_len - 1]
            labels = labels[:max_seq_len - 1]
        input_ids.append(sep_id)
        labels.append(unword_label_id)
        klen = len(input_ids)
        if klen < max_seq_len:
            pad_len = max_seq_len - klen
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [unword_label_id] * pad_len
        token_type_ids = [0] * max_seq_len
        attention_mask = [1] * klen + [0] * pad_len
        data2.append((np.asarray(input_ids, dtype="int64"), np.asarray(token_type_ids, dtype="int64"),
                      np.asarray(attention_mask, dtype="int64"), np.asarray(labels, dtype="int64")))
    return data2


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

simple_data = data[:15]
texts = []
labels = []
for dic in simple_data:
    texts.append(dic['text'])
    labels.append(dic['labels'])
print("训练数据：", len(simple_data), len(texts), len(labels))
flattened_list = np.concatenate(labels).tolist()
label_category = set(flattened_list)
print(label_category)

print(np.asarray(texts).shape)

# 创建数据加载器
batch_size = 16
num_workers = 1
# train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
# dev_dataloader = torch.utils.data.DataLoader(dev_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
label_num=len(label2id)+1
model=bert_model.BertModel(label_num)
