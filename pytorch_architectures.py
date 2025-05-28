import torch
import torch.nn as nn

class PytorchSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=128, hidden_size=256, tgt_vocab=None):  # 新增tgt_vocab参数
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
        self.tgt_vocab = tgt_vocab  # 保存目标词表

    def forward(self, x_enc, x_dec=None, max_decode_len=50):
        # 编码器
        x_enc_embed = self.embedding(x_enc)
        _, (h_enc, c_enc) = self.encoder(x_enc_embed)  # 仅保留最终状态

        # 解码器（训练时用教师强制，推理时自回归）
        if x_dec is not None:
            x_dec_embed = self.embedding(x_dec)
            h_dec, _ = self.decoder(x_dec_embed, (h_enc, c_enc))
            output = self.fc(h_dec)
        else:
            # 推理时逐词生成（简化示例）
            batch_size = x_enc.size(0)
            # 原代码：current_token = torch.full((batch_size, 1), zh_vocab["<SOS>"], dtype=torch.long)
            current_token = torch.full((batch_size, 1), self.tgt_vocab["<SOS>"], dtype=torch.long)  # 使用实例变量
            outputs = []
            h, c = h_enc, c_enc
            for _ in range(max_decode_len):
                x_embed = self.embedding(current_token)
                h, (h, c) = self.decoder(x_embed, (h, c))
                pred = self.fc(h)
                outputs.append(pred)
                current_token = pred.argmax(-1)  # 贪心搜索
            output = torch.cat(outputs, dim=1)
        return output