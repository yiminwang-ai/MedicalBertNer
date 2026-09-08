import jieba


def is_end_punctuation(word):
    # 定义常见的句子结束标点
    sentence_endings = ['。', '！', '？', '；', '…','，']
    return word in sentence_endings


def split_sentences_with_disambiguation(text):
    # 使用jieba进行分词
    words = list(jieba.cut(text))

    sentences = []
    current_sentence = []



    for word in words:
        current_sentence.append(word)

        # 检查是否是句子结束符
        if is_end_punctuation(word):
            # 在特定情况下，不将当前句子加入，如下：
            if len(current_sentence) < 3:  # 如果句子很短，不算
                continue

            sentences.append(''.join(current_sentence).strip())
            current_sentence = []  # 清空当前句子

    # 添加最后一个句子（如果有）
    if current_sentence:
        sentences.append(''.join(current_sentence).strip())

    return sentences

text1 = "我肚子天天叫，很胀啊。而且肠胃难受"
# 调用函数
sentences = split_sentences_with_disambiguation(text1)
print(sentences)

# 示例文本
text2 = "我肚子天天叫个不停很胀啊而且肠胃难受"

# 调用函数
sentences = split_sentences_with_disambiguation(text2)
print(sentences)
