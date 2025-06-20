1. 数据集说明
数据集采用了联合国大会平行语料库（中文-英文），数据集放在了data文件目录下面，由于时间关系和机器性能，选用数据集较小的devset和testset作为训练数据集和验证数据集。

2. 项目结构
--data文件目录（存放数据集和生成的预处理文件、词汇表）
--train.py 项目主函数入口，训练入口
--activation_functions.py 激活函数
--data_preprocessing.py 数据处理文件，生成词汇表等
-- lstm_layer.py lstm层实现
-- pytorch_architectures.py pytorch版本实现
-- rnn_architectures.py rnn层架构实现（一对一和seq2seq实现）

二、实验步骤
1. 数据预处理：平行语料清洗 对 UNv1.0.devset.en （英文）和 UNv1.0.devset.zh （中文）进行清洗，包括去除乱码、统一标点格式、过滤过短/过长句子对。
2. 分词与词表构建
   - 英文：按空格分词，处理缩写（如 "don't" → "do" + "n't"）；
   - 中文：使用分词工具（如jieba）进行分词；
   - 统计两种语言的词频，构建源语言（英文）和目标语言（中文）的词表（Vocabulary），并添加 <PAD> （填充）、 <UNK> （未登录词）、 <SOS> （句子开始）、 <EOS> （句子结束）特殊标记。
3. 序列向量化 将清洗后的文本序列转换为词表索引序列，对长度不一致的句子进行填充（如统一到最大长度或设定固定长度），生成训练用的输入（英文索引序列）和输出（中文索引序列）。
4. 激活函数库开发 实现 tanh 、 LeakyReLU 、 ELU 及其导数函数（用于反向传播），支持通过参数选择激活函数。
5. LSTM层前向传播（基础版） 使用NumPy实现单时间步LSTM单元的前向计算，公式包括：
   - 输入门 i_t = σ(W_ix * x_t + W_ih * h_{t-1} + b_i) ；
   - 遗忘门 f_t = σ(W_fx * x_t + W_fh * h_{t-1} + b_f) ；
   - 候选记忆细胞 c̃_t = activation_c(W_cx * x_t + W_ch * h_{t-1} + b_c) （activation_c可配置为tanh等）；
   - 记忆细胞 c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t ；
   - 输出门 o_t = σ(W_ox * x_t + W_oh * h_{t-1} + b_o) ；
   - 隐藏状态 h_t = o_t ⊙ activation_h(c_t) （activation_h可配置为tanh等）。 注：初始隐藏状态 h_0 和记忆细胞 c_0 初始化为零向量。
6. LSTM层反向传播 推导LSTM的梯度公式，计算各参数（ W_ix, W_ih, b_i 等）的梯度，包括时间步的反向传播（BPTT）。关键步骤：
   - 从输出层误差反向计算隐藏状态梯度 dh_t ；
   - 计算记忆细胞梯度 dc_t ，并向时间步 t-1 传递；
   - 计算各门控的梯度，最终得到权重矩阵和偏置的梯度。
7.实现Adam优化器（或SGD+动量），根据反向传播得到的梯度更新LSTM参数。
8. 向量化优化 将单时间步的循环计算转换为批量矩阵运算：
   - 合并时间步维度，将输入序列 X转换为矩阵运算；
   - 利用NumPy的广播机制，同时计算所有时间步的门控值和状态，避免显式循环（如用 np.dot 替代 for t in range(seq_len) ）。
9. RNN架构实现
   - 一对一（One-to-One） ：输出分类结果（适用于文本分类）。
   - 多对多（Seq2Seq） ：编码器处理输入序列（输出最终隐藏状态 h_enc ），解码器以 h_enc 为初始状态，逐时间步生成输出序列（需处理 <SOS> 起始符和 <EOS> 终止符）。
10. 编码器-解码器架构集成
    - 编码器：使用LSTM层处理输入英文序列，输出最终隐藏状态 h_enc ；
    - 解码器：以 h_enc 为初始状态，逐时间步生成中文序列（每个时间步输入前一时刻的预测词索引，通过嵌入层转换为向量后输入LSTM）；
    - 输出层：解码器的隐藏状态通过全连接层和softmax，预测当前时间步的词概率。
11. 训练流程优化
    - 批量训练（Batch Training） ：将数据按批次输入模型，计算批次损失（如交叉熵损失）；
    - 梯度裁剪（Gradient Clipping） ：对梯度范数超过阈值的参数梯度进行缩放，防止梯度爆炸；
    - 早停（Early Stopping） ：监控验证集的BLEU分数或损失，若连续若干轮无提升则停止训练。）
12. BLEU分数计算 实现BLEU评估指标：对模型生成的中文翻译序列与参考序列，计算n-gram匹配度（通常n=4），
13. 模型调优
    - 超参数调整（如学习率、隐藏层大小、词嵌入维度）；
    - 激活函数选择（测试tanh、LeakyReLU对LSTM性能的影响）；
- 数据增强（如回译法，用模型生成更多训练数据）。

