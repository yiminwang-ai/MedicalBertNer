import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
import numpy as np


class NERDataset(Dataset):
    def __init__(self, texts, labels, label_dict, verbose=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('BertModel')
        self.label_dict = label_dict
        self.verbose = verbose

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        max_len = 64
        # 标记化文本
        # 输入整句话
        # 标记化文本
        encoding = self.tokenizer(
            text,
            is_split_into_words=False,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_len,
            add_special_tokens=True
        )

        # 解码 token ids
        input_ids = encoding['input_ids'].squeeze().tolist()  # 将张量转换为列表
        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)  # 解码并跳过特殊标记
        label_ids = [self.label_dict['O']] + [self.label_dict[lbl] for lbl in label] + [self.label_dict['[PAD]']] * (
                max_len - len(label) - 1)
        if self.verbose:
            print("text", text, "label", label)
            print('inputids', encoding['input_ids'], np.asarray(encoding['input_ids']).shape)
            print('label_ids', label_ids, len(label_ids))
        label_ids = label_ids[:max_len]
        encoding['labels'] = torch.tensor(label_ids)
        #降维
        return {key: val.squeeze() for key, val in encoding.items()}
