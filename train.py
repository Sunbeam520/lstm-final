import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import nn

from pytorch_architectures import PytorchSeq2Seq
from rnn_architectures import Seq2Seq
import time

def evaluate(custom_model, val_src, val_tgt, tgt_vocab):
    """评估自定义模型翻译效果（计算BLEU分数）"""
    smooth = SmoothingFunction().method4
    total_bleu = 0.0
    idx2word = {v: k for k, v in tgt_vocab.items()}
    
    for src, tgt in zip(val_src, val_tgt):
        # 生成翻译结果（词列表）
        translated = custom_model.translate(np.array([src]))  # 假设translate方法已实现
        
        # 参考序列（真实标签，过滤特殊标记）
        reference = []
        for idx in tgt:
            word = idx2word.get(idx, "<UNK>")
            if word in ["<SOS>", "<EOS>", "<PAD>"]:
                continue
            reference.append(word)
        
        # 计算BLEU（至少需要1个参考词）
        if len(reference) > 0 and len(translated) > 0:
            bleu = sentence_bleu([reference], translated, smoothing_function=smooth)
            total_bleu += bleu
    
    return total_bleu / len(val_src)  # 平均BLEU分数

def evaluate_pytorch(pytorch_model, val_src, val_tgt, tgt_vocab):
    """评估PyTorch模型翻译效果（计算BLEU分数）"""
    smooth = SmoothingFunction().method4
    total_bleu = 0.0
    idx2word = {v: k for k, v in tgt_vocab.items()}
    
    for src, tgt in zip(val_src, val_tgt):
        # 准备输入张量（batch_size=1）
        # 原代码：src_tensor = torch.tensor([src], dtype=torch.long)
        src_tensor = torch.tensor(np.array([src]), dtype=torch.long)  # 优化列表转张量方式
        
        # 模型推理生成翻译结果（禁用梯度）
        with torch.no_grad():
            outputs = pytorch_model(src_tensor)  # 推理时x_dec为None，触发自回归生成
        
        # 从输出logits中提取词索引（形状：[seq_len]）
        pred_indices = outputs.argmax(dim=-1).squeeze().tolist()
        
        # 转换为词列表（过滤特殊标记）
        translated = []
        for idx in pred_indices:
            word = idx2word.get(idx, "<UNK>")
            if word in ["<SOS>", "<EOS>", "<PAD>"]:
                continue
            translated.append(word)
        
        # 处理参考序列（真实标签，过滤特殊标记）
        reference = []
        for idx in tgt:
            word = idx2word.get(idx, "<UNK>")
            if word in ["<SOS>", "<EOS>", "<PAD>"]:
                continue
            reference.append(word)
        
        # 计算BLEU分数（至少需要1个参考词和1个翻译词）
        if len(reference) > 0 and len(translated) > 0:
            bleu = sentence_bleu([reference], translated, smoothing_function=smooth)
            total_bleu += bleu
    
    return total_bleu / len(val_src)  # 平均BLEU分数

def main():
    # 超参数设置
    embedding_dim = 128
    hidden_size = 64
    learning_rate = 0.001
    epochs = 5
    batch_size = 2

    # 加载数据
    en_vectors = np.load("data/en_vectors.npy")
    zh_vectors = np.load("data/zh_vectors.npy")
    val_en_vectors = np.load("data/val_en_vectors.npy")
    val_zh_vectors = np.load("data/val_zh_vectors.npy")
    en_vocab = np.load("data/en_vocab.npy", allow_pickle=True).item()
    zh_vocab = np.load("data/zh_vocab.npy", allow_pickle=True).item()

    # 初始化自定义模型
    custom_model = Seq2Seq(
        input_size=embedding_dim,
        hidden_size=hidden_size,
        output_vocab_size=len(zh_vocab),
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(zh_vocab),
        embedding_dim=embedding_dim,
        tgt_vocab=zh_vocab,
        teacher_forcing_ratio=0.5
    )

    # 训练自定义模型
    custom_train_times = []
    num_samples = len(en_vectors)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for i in range(0, num_samples, batch_size):
            src_batch = en_vectors[i:i+batch_size]
            tgt_batch = zh_vectors[i:i+batch_size]
            loss = custom_model.train_step(src_batch, tgt_batch, learning_rate)
            total_loss += loss
        
        epoch_time = time.time() - start_time
        custom_train_times.append(epoch_time)
        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Custom Model Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: {avg_loss:.4f}")

        # 每轮训练后评估
        avg_bleu = evaluate(custom_model, val_en_vectors, val_zh_vectors, zh_vocab)
        print(f"Custom Model Epoch {epoch+1}, Average BLEU Score: {avg_bleu:.4f}")

    print(f"自定义模型平均每轮训练时间: {np.mean(custom_train_times):.2f}s")

def train_pytorch_model(epochs, batch_size):  # 新增batch_size参数
    # 加载数据
    en_vectors = torch.tensor(np.load("data/en_vectors.npy"), dtype=torch.long)
    zh_vectors = torch.tensor(np.load("data/zh_vectors.npy"), dtype=torch.long)
    en_vocab = np.load("data/en_vocab.npy", allow_pickle=True).item()
    zh_vocab = np.load("data/zh_vocab.npy", allow_pickle=True).item()
    val_en_vectors = np.load("data/val_en_vectors.npy")
    val_zh_vectors = np.load("data/val_zh_vectors.npy")

    # 初始化PyTorch模型（新增tgt_vocab参数）
    pytorch_model = PytorchSeq2Seq(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(zh_vocab),
        embedding_dim=128,
        hidden_size=256,
        tgt_vocab=zh_vocab  # 传递目标词表
    )
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab["<PAD>"])

    # 训练PyTorch模型
    pytorch_train_times = []
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        pytorch_model.train()
        
        for i in range(0, len(en_vectors), batch_size):  # 使用传入的batch_size
            src_batch = en_vectors[i:i+batch_size]
            tgt_batch = zh_vectors[i:i+batch_size]
            tgt_input = tgt_batch[:, :-1]  # 解码器输入（去掉EOS）
            tgt_labels = tgt_batch[:, 1:]  # 解码器标签（去掉SOS）

            optimizer.zero_grad()
            outputs = pytorch_model(src_batch, tgt_input)
            loss = criterion(outputs.reshape(-1, len(zh_vocab)), tgt_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        pytorch_train_times.append(epoch_time)
        avg_loss = total_loss / (len(en_vectors) // batch_size)
        print(f"PyTorch Model Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: {avg_loss:.4f}")

        # 每轮训练后评估（调用新增的evaluate_pytorch）
        avg_bleu = evaluate_pytorch(pytorch_model, val_en_vectors, val_zh_vectors, zh_vocab)
        print(f"PyTorch Model Epoch {epoch+1}, Average BLEU Score: {avg_bleu:.4f}")

    print(f"PyTorch模型平均每轮训练时间: {np.mean(pytorch_train_times):.2f}s")

if __name__ == "__main__":
    main()
    train_pytorch_model(epochs=5, batch_size=2)  # 调用新的训练函数