import re
import jieba
from collections import defaultdict

import numpy as np

# ==================== 超参数配置 ====================
MIN_SENT_LENGTH = 3    # 句子最小长度（词数）
MAX_SENT_LENGTH = 50   # 句子最大长度（词数）
MIN_WORD_FREQ = 2      # 词表最小词频（低于此频率的词用<UNK>代替）
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# ==================== 工具函数 ====================
def clean_text(text, lang):
    """
    清洗文本：去除乱码、统一标点格式
    :param text: 输入文本（单句）
    :param lang: 语言类型（'en'或'zh'）
    :return: 清洗后的文本
    """
    # 去非打印字符（乱码）
    cleaned = re.sub(r'[^\x20-\x7E\u4E00-\u9FFF]', ' ', text)  # 保留英文可打印字符和中文字符
    # 统一标点（英文半角，中文全角）
    if lang == 'en':
        cleaned = re.sub(r'[‘’“”]', "'", cleaned)  # 英文引号转半角
        cleaned = re.sub(r'[，。！？]', ',', cleaned)  # 中文标点转英文（根据需求调整）
    elif lang == 'zh':
        cleaned = re.sub(r'[,\.!?]', '，。！？', cleaned)  # 英文标点转中文（根据需求调整）
    # 去除多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def tokenize_en(sentence):
    """
    英文分词（处理缩写）
    :param sentence: 清洗后的英文句子
    :return: 分词列表
    """
    # 处理常见缩写（如 don't → do n't）
    pattern = r"\b(can't|won't|n't|'s|'re|'ve|'ll|'d)\b"
    processed = re.sub(pattern, r" \1 ", sentence)
    # 按空格分词（已处理过连续空格）
    tokens = processed.split()
    return tokens

def tokenize_zh(sentence):
    """
    中文分词（使用jieba）
    :param sentence: 清洗后的中文句子
    :return: 分词列表
    """
    return list(jieba.cut(sentence))

def build_vocab(tokenized_corpus, min_freq=MIN_WORD_FREQ):
    """
    构建词表（含特殊标记）
    :param tokenized_corpus: 分词后的语料（二维列表，[[token1, token2...], ...]）
    :param min_freq: 最小词频
    :return: 词到索引的映射字典
    """
    word_freq = defaultdict(int)
    for tokens in tokenized_corpus:
        for token in tokens:
            word_freq[token] += 1
    
    # 过滤低频词，添加特殊标记（按优先级排序）
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        SOS_TOKEN: 2,
        EOS_TOKEN: 3
    }
    current_idx = 4
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq and word not in vocab:
            vocab[word] = current_idx
            current_idx += 1
    return vocab

def vectorize_sequence(tokens, vocab, max_length):
    """
    将分词序列转换为索引序列（含填充和截断）
    :param tokens: 分词列表
    :param vocab: 词表字典
    :param max_length: 最大序列长度
    :return: 索引列表（长度为max_length）
    """
    # 添加SOS和EOS标记
    indexed = [vocab[SOS_TOKEN]] + [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens] + [vocab[EOS_TOKEN]]
    # 截断或填充
    if len(indexed) > max_length:
        indexed = indexed[:max_length]
    else:
        indexed += [vocab[PAD_TOKEN]] * (max_length - len(indexed))
    return indexed

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 读取并清洗语料
    en_path = r"c:\Users\57617\PycharmProjects\lstm-final\data\UNv1.0.devset.en"
    zh_path = r"c:\Users\57617\PycharmProjects\lstm-final\data\UNv1.0.devset.zh"
    
    try:
        with open(en_path, 'r', encoding='utf-8') as f:
            en_sentences = [line.strip() for line in f.readlines()]
        with open(zh_path, 'r', encoding='utf-8') as f:
            zh_sentences = [line.strip() for line in f.readlines()]
    except FileNotFoundError as e:
        raise RuntimeError(f"语料文件未找到：{e}")
    
    # 检查句子对对齐
    if len(en_sentences) != len(zh_sentences):
        raise RuntimeError(f"英文（{len(en_sentences)}句）与中文（{len(zh_sentences)}句）语料行数不一致")
    
    # 清洗并分词
    en_tokenized = []
    zh_tokenized = []
    valid_pairs = []  # 保存有效句子对索引
    
    for i in range(len(en_sentences)):
        en_clean = clean_text(en_sentences[i], 'en')
        zh_clean = clean_text(zh_sentences[i], 'zh')
        
        en_tokens = tokenize_en(en_clean)
        zh_tokens = tokenize_zh(zh_clean)
        
        # 过滤过短/过长句子对
        if len(en_tokens) < MIN_SENT_LENGTH or len(en_tokens) > MAX_SENT_LENGTH:
            continue
        if len(zh_tokens) < MIN_SENT_LENGTH or len(zh_tokens) > MAX_SENT_LENGTH:
            continue
        
        en_tokenized.append(en_tokens)
        zh_tokenized.append(zh_tokens)
        valid_pairs.append(i)
    
    print(f"[调试] 原始句子对数量：{len(en_sentences)}，清洗后有效句子对数量：{len(valid_pairs)}")
    print(f"[调试] 示例英文分词：{en_tokenized[0][:5]}...")  # 打印前5个词
    print(f"[调试] 示例中文分词：{zh_tokenized[0][:5]}...")

    # 2. 构建词表
    en_vocab = build_vocab(en_tokenized)
    zh_vocab = build_vocab(zh_tokenized)
    print(f"[调试] 英文词表大小：{len(en_vocab)}（含4个特殊标记）")
    print(f"[调试] 中文词表大小：{len(zh_vocab)}（含4个特殊标记）")
    print(f"[调试] 英文词表前5项：{list(en_vocab.items())[:5]}")
    print(f"[调试] 中文词表前5项：{list(zh_vocab.items())[:5]}")

    # 3. 序列向量化（计算最大长度）
    max_en_length = max(len(tokens) + 2 for tokens in en_tokenized)  # +2为SOS和EOS
    max_zh_length = max(len(tokens) + 2 for tokens in zh_tokenized)
    max_length = max(max_en_length, max_zh_length, MAX_SENT_LENGTH + 2)  # 取最大长度
    
    en_vectors = [vectorize_sequence(tokens, en_vocab, max_length) for tokens in en_tokenized]
    zh_vectors = [vectorize_sequence(tokens, zh_vocab, max_length) for tokens in zh_tokenized]
    
    print(f"[调试] 向量化后英文序列示例：{en_vectors[0][:10]}...（长度：{len(en_vectors[0])}）")
    print(f"[调试] 向量化后中文序列示例：{zh_vectors[0][:10]}...（长度：{len(zh_vectors[0])}）")

    # 可选：保存预处理结果（根据需求调整）

    np.save("data/en_vectors.npy", np.array(en_vectors))
    np.save("data/zh_vectors.npy", np.array(zh_vectors))
    # 在数据预处理脚本末尾添加保存词表的代码（示例）
    np.save("data/en_vocab.npy", en_vocab)  # en_vocab 是字典
    np.save("data/zh_vocab.npy", zh_vocab)  # zh_vocab 是字典
