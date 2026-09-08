import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_path = r"D:\Training\MedicalBertNer\BertModel"
# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def get_embedding(text):
    # 将文本编码为BERT输入格式
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取[CLS] token的隐藏状态
    return outputs.last_hidden_state[:, 0, :].numpy()  # 使用[CLS]的表示


def calculate_similarity(text1, text2):

    # 获取文本的嵌入
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)

    # 计算余弦相似度
    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    # 输出结果
    print(f"文本1: \"{text1}\"")
    print(f"文本2: \"{text2}\"")
    print(f"余弦相似度: {similarity:.4f}")
    return similarity
calculate_similarity("肚子咕噜咕噜的叫","肠鸣")


import nmslib

# 生成一些示例数据，假设这是你的标准集合 S 的嵌入表示
# 例如：每个医学实体的嵌入表示为一个 128 维的向量
standard_set_embeddings = np.random.rand(1000, 128).astype(np.float32)  # 1000 个实体的嵌入向量

# 创建 NMSLIB 的 HNSW 索引
hnsw_index = nmslib.init(method='hnsw', space='l2')

# 添加标准集合 S 到索引中
hnsw_index.addDataPointBatch(standard_set_embeddings)

# 创建索引
hnsw_index.createIndex()

# 一个新的文本的嵌入表示
query_embedding = np.random.rand(1, 128).astype(np.float32)  # 这里你应该用你自己的文本计算的嵌入表示

# 搜索 K 个最近邻，设定距离度量
K = 5  # 查找最近的 5 个邻居
neighbors = hnsw_index.knnQuery(query_embedding, k=K)

# 输出结果
neighbor_ids, distances = neighbors
print("Nearest Neighbors IDs:", neighbor_ids)
print("Distances:", distances)
