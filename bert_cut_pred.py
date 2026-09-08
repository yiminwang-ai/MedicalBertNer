import numpy as np
import torch
from transformers import BertTokenizer, BertForTokenClassification

import bert_cut_train

# 假设你已经训练过的模型路径
model_path = './MNER_Model'  # 替换为你的模型路径
original_model_path = './BertModel'  # 替换为你的模型路径
# 加载模型和标记器
model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('BertModel')  # 使用与你训练时相同的tokenizer

# 设置为评估模式
model.eval()


label_mapping = bert_cut_train.label_dict # 修改此行，确保正确建立映射
# 现在我们创建标签列表
label_list = [lbl for lbl, idx in label_mapping.items()]
print('label_mapping', label_mapping)

def predict_entities(text, label_mapping):
    # 标记化文本
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model(**encoding)

    # 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    print(predictions)
    # 将预测转换为标签
    predicted_labels = predictions[0].tolist()
    print("predict", predicted_labels)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0].tolist())
    print(tokens)

    results = []
    for token, pred in zip(tokens, predicted_labels):
        if pred < len(label_mapping):
            results.append((token, label_list[pred]))
        else:
            results.append((token, "Unknown Label"))

    # 返回预测结果
    return results

while True:
    entities=[]
    # 测试新的文本段落
    new_text = input("请你输入：")
    predicted_entities = predict_entities(new_text, label_mapping)
    entity_token = ''
    entity_label = []
    # 打印结果
    for token, label in predicted_entities:

        if label.startswith('B'):
            if len(entity_token)>0:
                entities.append({entity_token:entity_label})
                entity_token = ''
                entity_label = []

            entity_token += token
            entity_label.append(label)
        else:
            if label.startswith('I'):
                entity_token+=token
                entity_label.append(label)
    refined_entities = []
    for e in entities:
        # 获取 tokens 和 labels
        tokens = list(e.keys())  # 将 tokens 变为列表
        lbl = list(e.values())  # 将 lbl 变为列表
        lbl=np.asarray(lbl).squeeze()

        # 确保 lbl 不为空
        if len(lbl)>0:
            token_label = lbl[0].split('-')[1]  # 获取第一个 label 的第二部分 (sym, mic, 等)
            refined_entities.append({tokens[0]: token_label})  # 将 tokens[0] 和 token_label 作为字典项添加

    # 打印 refined_entities
    print(refined_entities)