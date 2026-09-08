# MedicalBertNer

Medical Named Entity Recognition (NER) based on **BERT + BiLSTM + CRF**.

## Introduction

This project implements a medical NER pipeline using BERT for contextual token embeddings, BiLSTM for bidirectional sequence modeling, and CRF for globally optimized tag decoding. It is designed for extracting clinical entities from **Chinese medical texts** (e.g., electronic health records, discharge summaries).

## Model Architecture

Raw Text
  ↓
Tokenizer (BERT WordPiece)
  ↓
BERT → Contextual Token Embeddings
  ↓
BiLSTM → Bidirectional Sequence Features
  ↓
CRF → Global Tag Decoding (Viterbi)
  ↓
BIO / BIOES Label Sequence

## Entity Types

| Tag | Description | Example |
|-----|-------------|---------|
| `Disease` | 疾病 | 2型糖尿病 (Type 2 Diabetes) |
| `Symptom` | 症状 | 头痛 (Headache) |
| `Drug` | 药物 | 二甲双胍 (Metformin) |
| `Test` | 检查 | CT (Computed Tomography) |
| `Body` | 部位 | 左肺 (Left Lung) |

## Environment

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
