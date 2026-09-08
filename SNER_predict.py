import torch
import json
from transformers import BertTokenizer

# 标签
label_dict = {'O': 0, 'B-dis': 1, 'I-dis': 2, 'B-sym': 3, 'I-sym': 4, 'B-mic': 5, 'I-mic': 6, 'B-ite': 7,
              'I-ite': 8, 'B-pro': 9, 'I-pro': 10, 'B-dep': 11, "I-dep": 12, 'B-dru': 13,
              'I-dru': 14, 'B-bod': 15, 'I-bod': 16, '[PAD]': 17,'[CLS]': 18, '[SEP]': 19, "[UNK]": 20}
# 加载保存的模型
model_path = './MNER_Bert_BiLSTM_CRF_Model/best_model.pth'  # 你的模型保存路径

# 加载模型
model = torch.load(model_path)
model.eval()  # 设置为评估模式

# 移动模型到适当的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('./BertModel')


def predict(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 获取模型的预测
    with torch.no_grad():  # 不需要计算梯度
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # 将 logits 转换为标签
    preds = model.crf.decode(logits)

    # 转换索引到标签
    label_list = list(label_dict.keys())
    output = []
    for pred in preds:
        output.append([label_list[i] for i in pred])

    return output


# 测试文本
test_texts = ["我感觉喉咙难受，想吐，胃胀"]

# 进行预测
predictions = predict(test_texts)

# 输出结果
for text, preds in zip(test_texts, predictions):
    print(f"文本: {text}")
    print(f"预测标签: {preds}")
