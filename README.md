# MedicalBertNer
# MedicalBertNer

基于 **BERT + BiLSTM + CRF** 的医学实体识别（Medical Named Entity Recognition）。

## 简介

利用 BERT 预训练语言模型提取上下文特征，经 BiLSTM 捕获序列双向依赖，最后由 CRF 进行全局标签解码，实现医学文本中的实体识别（如疾病、症状、药物、检查等）。

## 模型结构

输入文本
  ↓
BERT (预训练语言模型，提取字/词级上下文特征)
  ↓
BiLSTM (双向 LSTM，捕获前后文序列信息)
  ↓
CRF (条件随机场，约束标签转移，全局最优解码)
  ↓
实体标签序列

## 环境

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn

bash
pip install torch transformers scikit-learn


## 使用

bash
# 训练
python train.py --data_dir ./data --model_name bert-base-chinese

# 预测
python predict.py --input "患者诊断为2型糖尿病，需服用二甲双胍"


## 输入 / 输出

- 输入：医学文本（中文 / 英文）
- 输出：实体及类别（BIO / BIOES 标注格式）

示例：

输入：患者诊断为2型糖尿病，需服用二甲双胍
输出：
2型糖尿病 → Disease
二甲双胍 → Drug


## 应用方向

- 🏥 电子病历结构化
- 📄 医学文献信息抽取
- 🤖 临床决策支持系统
- 🔬 医学知识图谱构建

## License

Apache-2.0
