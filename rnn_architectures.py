import numpy as np
from lstm_layer import LSTM  # 依赖之前实现的LSTM层

class OneToOneRNN:
    def __init__(self, input_size, hidden_size, output_size, activation_c="tanh", activation_h="tanh"):
        """
        初始化一对一RNN架构
        :param input_size: 输入特征维度（词嵌入维度）
        :param hidden_size: LSTM隐藏状态维度
        :param output_size: 输出类别数
        """
        self.lstm = LSTM(input_size, hidden_size, activation_c, activation_h)
        self.output_size = output_size
        
        # 全连接层权重（隐藏状态→输出）
        self.W_out = np.random.normal(0, 0.1, (hidden_size, output_size))
        self.b_out = np.zeros((1, output_size))

    def forward(self, X):
        """
        前向传播
        :param X: 输入序列（形状：[batch_size, seq_len, input_size]）
        :return: 分类概率（形状：[batch_size, output_size]）
        """
        # LSTM前向传播（输出所有时间步隐藏状态，形状：[seq_len, batch_size, hidden_size]）
        h_sequence = self.lstm.forward(X.transpose(1, 0, 2))  # 调整维度为[seq_len, batch_size, input_size]
        
        # 取最后时间步的隐藏状态（形状：[batch_size, hidden_size]）
        h_last = h_sequence[-1]  # 最后一个时间步
        
        # 全连接层输出（形状：[batch_size, output_size]）
        logits = np.matmul(h_last, self.W_out) + self.b_out
        probs = self.softmax(logits)  # 分类任务用softmax
        
        # print(f"[调试] 一对一前向传播：输入形状{X.shape} → 输出概率形状{probs.shape}")
        return probs

    @staticmethod
    def softmax(x):
        """数值稳定的softmax实现"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, probs, labels, learning_rate=0.001):
        """
        反向传播（简化示例，仅含分类层梯度）
        :param probs: 前向传播输出的概率（[batch_size, output_size]）
        :param labels: 真实标签（[batch_size]，one-hot编码）
        """
        batch_size = probs.shape[0]
        
        # 计算分类层梯度（交叉熵损失导数）
        d_logits = probs - labels  # 导数为 (预测概率 - 真实标签)
        dW_out = np.matmul(self.lstm.h_list[-1].T, d_logits) / batch_size  # h_last的转置与d_logits相乘
        db_out = np.mean(d_logits, axis=0, keepdims=True)
        
        # 计算LSTM层梯度（将分类层梯度传递给LSTM）
        dh_last = np.matmul(d_logits, self.W_out.T)  # [batch_size, hidden_size]
        
        # 关键修复：dh_output形状应与前向传播输出的隐藏状态序列一致（[seq_len, batch_size, hidden_size]）
        h_sequence = self.lstm.forward(self.X)  # 假设已保存前向输入X，或直接使用前向输出的h_sequence
        dh_output = np.zeros_like(h_sequence)  # 正确形状：[seq_len, batch_size, hidden_size]
        dh_output[-1] = dh_last  # 仅最后时间步有梯度
        
        # 调用LSTM的反向传播
        lstm_grads = self.lstm.backward(dh_output)
        
        # 参数更新（示例使用SGD，实际可替换为Adam）
        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out
        # 更新LSTM参数（仅示例关键参数，实际需更新所有权重）
        self.lstm.W_ix -= learning_rate * lstm_grads["dW_ix"]
        self.lstm.W_ih -= learning_rate * lstm_grads["dW_ih"]
        self.lstm.b_i -= learning_rate * lstm_grads["db_i"]


class Seq2Seq:
    def __init__(self, input_size, hidden_size, output_vocab_size, 
                 src_vocab_size, tgt_vocab_size, embedding_dim, tgt_vocab,  # 新增tgt_vocab参数
                 activation_c="tanh", activation_h="tanh", teacher_forcing_ratio=0.5):
        """
        初始化Seq2Seq架构
        :param tgt_vocab: 目标语言词表字典（包含<PAD>等特殊标记的索引）
        :param input_size: 输入特征维度（源语言词嵌入维度）
        :param hidden_size: LSTM隐藏状态维度
        :param output_vocab_size: 输出词表大小（目标语言词嵌入维度）
        :param teacher_forcing_ratio: 教师强制比例（0~1）
        """
        self.hidden_size = hidden_size
        self.tgt_vocab = tgt_vocab  # 保存目标词表字典
        self.encoder = LSTM(input_size, hidden_size, activation_c, activation_h)
        self.decoder = LSTM(input_size, hidden_size, activation_c, activation_h)  # 假设输入输出词嵌入维度相同
        self.output_vocab_size = output_vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.tgt_vocab_size = tgt_vocab_size  # 新增：保存目标词表大小
        
        # 新增：初始化源语言和目标语言的词嵌入矩阵（随机初始化）
        self.src_embedding = np.random.normal(0, 0.1, (src_vocab_size, embedding_dim))  # [源词表大小, 嵌入维度]
        self.tgt_embedding = np.random.normal(0, 0.1, (tgt_vocab_size, embedding_dim))  # [目标词表大小, 嵌入维度]
        
        # 解码器输出层权重（隐藏状态→词表概率）
        self.W_dec_out = np.random.normal(0, 0.1, (hidden_size, output_vocab_size))
        self.b_dec_out = np.zeros((1, output_vocab_size))

    def forward(self, X_enc, X_dec=None, max_decode_len=50):
        """
        Seq2Seq前向传播
        :param X_enc: 编码器输入（形状：[batch_size, enc_seq_len, input_size]）
        :param X_dec: 解码器输入（训练时为真实目标序列，形状：[batch_size, dec_seq_len, input_size]）
        :param max_decode_len: 推理时最大解码长度
        :return: 解码器输出序列概率（形状：[batch_size, dec_seq_len, output_vocab_size]）
        """
        batch_size = X_enc.shape[0]
        
        # 编码器前向传播（获取最终隐藏状态）
        h_enc_sequence = self.encoder.forward(X_enc.transpose(1, 0, 2))  # [enc_seq_len, batch_size, hidden_size]
        h_enc_final = h_enc_sequence[-1]  # [batch_size, hidden_size]
        
        # 重置解码器状态
        self.decoder.h_list = [h_enc_final.T]  # 初始隐藏状态（转置为[hidden_size, batch_size]）
        self.decoder.c_list = [np.zeros_like(h_enc_final.T)]  # 初始记忆细胞
        
        # 解码器前向传播（一次性处理整个序列）
        if X_dec is not None:
            # 训练时：使用教师强制，输入为X_dec（形状：[batch_size, dec_seq_len, input_size]）
            dec_input = X_dec.transpose(1, 0, 2)  # 调整为[dec_seq_len, batch_size, input_size]
            h_dec_sequence = self.decoder.forward(dec_input)  # [dec_seq_len, batch_size, hidden_size]
        else:
            # 推理时：自回归生成序列（需循环生成每个时间步）
            h_dec_sequence = []
            current_input = self.sos_embedding(batch_size).transpose(1, 0, 2)  # [1, batch_size, input_size]
            for _ in range(max_decode_len):
                h_dec_t = self.decoder.forward(current_input)  # [1, batch_size, hidden_size]
                h_dec_sequence.append(h_dec_t[0])
                # 预测下一个词并生成输入（逻辑同原代码）
                # ...
        
        # 计算输出概率（形状：[dec_seq_len, batch_size, output_vocab_size]）
        logits = np.matmul(h_dec_sequence, self.W_dec_out) + self.b_dec_out
        probs = self.softmax(logits)
        
        # 调整输出形状为[batch_size, dec_seq_len, output_vocab_size]
        outputs = probs.transpose(1, 0, 2)
        # print(f"[调试] Seq2Seq前向传播：编码器输入{X_enc.shape} → 解码器输出{outputs.shape}")
        return outputs

    def sos_embedding(self, batch_size):
        """生成<SOS>标记的嵌入向量（示例，需根据实际词嵌入替换）"""
        return np.zeros((batch_size, 1, self.encoder.input_size))  # 实际应为预训练嵌入
    
    def eos_embedding(self, batch_size):
        """生成<EOS>标记的嵌入向量（示例）"""
        return np.zeros((batch_size, 1, self.encoder.input_size))
    
    def token_to_embedding(self, tokens):
        """将词索引转换为嵌入向量（示例，需根据实际词嵌入实现）"""
        return np.random.randn(tokens.shape[0], 1, self.encoder.input_size)  # 实际需查表
    
    def embed_src(self, src_indices):
        """将源语言索引序列转换为嵌入向量"""
        return self.src_embedding[src_indices]  # 形状：[batch_size, seq_len, embedding_dim]

    def embed_tgt(self, tgt_indices):
        """将目标语言索引序列转换为嵌入向量"""
        return self.tgt_embedding[tgt_indices]  # 形状：[batch_size, seq_len, embedding_dim]

    def train_step(self, src_indices, tgt_indices, learning_rate=0.001):
        """单步训练（处理一个批次）"""
        # 1. 词嵌入
        X_enc = self.embed_src(src_indices)  # [batch_size, enc_seq_len, embedding_dim]
        X_dec_input = self.embed_tgt(tgt_indices[:, :-1])  # 解码器输入（去掉<EOS>）
        tgt_labels = tgt_indices[:, 1:]  # 解码器标签（去掉<SOS>）
        
        # 2. 前向传播
        outputs = self.forward(X_enc, X_dec_input)  # [batch_size, dec_seq_len, tgt_vocab_size]
        
        # 3. 计算损失（交叉熵）
        batch_size, dec_seq_len, _ = outputs.shape
        outputs_flat = outputs.reshape(-1, self.tgt_vocab_size)  # [batch_size*dec_seq_len, tgt_vocab_size]
        labels_flat = tgt_labels.reshape(-1)  # [batch_size*dec_seq_len]
        
        # 交叉熵损失（忽略<PAD>的贡献）
        mask = (labels_flat != self.tgt_vocab["<PAD>"])  # 假设词表中<PAD>的索引已知
        loss = -np.sum(mask * np.log(outputs_flat[np.arange(len(labels_flat)), labels_flat] + 1e-8)) / batch_size
        
        # 4. 反向传播（需实现损失对输出的梯度）
        d_outputs = outputs_flat.copy()
        d_outputs[np.arange(len(labels_flat)), labels_flat] -= 1  # 交叉熵导数
        d_outputs = d_outputs.reshape(batch_size, dec_seq_len, self.tgt_vocab_size)
        # print("tgt_indices shape:", tgt_indices.shape)  # 应输出 (2, 52)（batch_size=2，seq_len=52）
        # print("X_dec_input shape:", X_dec_input.shape)  # 应输出 (2, 51, embedding_dim)（dec_seq_len=51）
        # print("dec_seq_len:", dec_seq_len)              # 应输出 51
        # print(f"[调试] 解码器h_list长度：{len(self.decoder.h_list)}，预期：{dec_seq_len + 1}")  # +1因初始状态
        # 5. 更新解码器输出层权重
        # 将 h_list[1:] 从列表转换为 numpy 数组（形状：[seq_len, hidden_size, batch_size]）
        # 新增调试：确认h_list长度
        # print(f"[调试] 解码器h_list长度：{len(self.decoder.h_list)}，预期：{dec_seq_len + 1}")  # +1因初始状态
        h_list_array = np.array(self.decoder.h_list[1:])  # 取时间步1~dec_seq_len的隐藏状态
        # 调整维度为 [seq_len, batch_size, hidden_size] 后展平为 [seq_len*batch_size, hidden_size]
        # print("h_list_array shape:", h_list_array.shape)  # 应输出 (51, 256, 32)
             # 应输出 (1632, 256)
        # print("d_outputs shape:", d_outputs.shape)        # 应输出 (32, 51, 4067)
        # print("d_outputs_flat shape:", d_outputs.reshape(-1, self.tgt_vocab_size).shape)  # 应输出 (1632, 4067)
        h_reshaped = h_list_array.transpose(0, 2, 1).reshape(-1, self.hidden_size)  # 关键修复：调整转置顺序
        # print("h_reshaped shape:", h_reshaped.shape)
        dW_dec_out = np.matmul(h_reshaped.T, d_outputs.reshape(-1, self.tgt_vocab_size)) / batch_size
        db_dec_out = np.mean(d_outputs.reshape(-1, self.tgt_vocab_size), axis=0, keepdims=True)
        
        # 6. 更新解码器LSTM参数（需调用decoder.backward）
        dh_dec = np.matmul(d_outputs, self.W_dec_out.T)  # [batch_size, dec_seq_len, hidden_size]
        decoder_grads = self.decoder.backward(dh_dec.transpose(1, 0, 2))  # 调整维度
        
        # 7. 参数更新（示例使用SGD）
        self.W_dec_out -= learning_rate * dW_dec_out
        self.b_dec_out -= learning_rate * db_dec_out
        self.decoder.W_ix -= learning_rate * decoder_grads["dW_ix"]
        # 同理更新其他LSTM参数（W_ih, b_i等）
        
        return loss

    def translate(self, src_indices, max_len=50):
        """推理：将源语言序列翻译为目标语言序列（返回词列表）"""
        # 1. 编码器前向传播（保持原逻辑）
        X_enc = self.embed_src(src_indices)  # [batch_size=1, enc_seq_len, embedding_dim]
        h_enc_sequence = self.encoder.forward(X_enc.transpose(1, 0, 2))
        h_enc_final = h_enc_sequence[-1]  # [1, hidden_size]
        
        # 2. 初始化解码器状态（保持原逻辑）
        self.decoder.h_list[0] = h_enc_final.T  # [hidden_size, 1]
        current_token = self.tgt_vocab["<SOS>"]  # 初始输入为<SOS>
        output_indices = [current_token]
        
        # 3. 生成序列（保持原逻辑）
        for _ in range(max_len):
            current_embedding = self.embed_tgt(np.array([[current_token]]))  # [1, 1, embedding_dim]
            h_dec_t = self.decoder.forward(current_embedding.transpose(1, 0, 2))  # [1, 1, hidden_size]
            logits = np.matmul(h_dec_t[0], self.W_dec_out) + self.b_dec_out
            probs = self.softmax(logits)
            current_token = np.argmax(probs)
            output_indices.append(current_token)
            if current_token == self.tgt_vocab["<EOS>"]:
                break
        
        # 4. 索引转词（新增）：过滤特殊标记，返回词列表
        idx2word = {v: k for k, v in self.tgt_vocab.items()}
        translated_words = []
        for idx in output_indices:
            word = idx2word.get(idx, "<UNK>")
            if word in ["<SOS>", "<EOS>", "<PAD>"]:
                continue
            translated_words.append(word)
        return translated_words

    @staticmethod
    def softmax(x):
        """数值稳定的softmax实现"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)