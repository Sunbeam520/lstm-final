import numpy as np
from activation_functions import ActivationFunctions

class LSTM:
    def __init__(self, input_size, hidden_size, activation_c="tanh", activation_h="tanh"):
        """
        初始化LSTM层
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏状态维度
        :param activation_c: 候选记忆细胞激活函数（默认tanh）
        :param activation_h: 隐藏状态激活函数（默认tanh）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 从ActivationFunctions获取激活函数及其导数
        self.activation_c, self.activation_c_deriv = ActivationFunctions.get_activation(activation_c)
        self.activation_h, self.activation_h_deriv = ActivationFunctions.get_activation(activation_h)

        # 初始化权重（示例，实际需根据公式调整）
        self.W_ix = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.W_ih = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_i = np.zeros((1, hidden_size))

        self.W_fx = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.W_fh = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_f = np.zeros((1, hidden_size))

        self.W_ox = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.W_oh = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_o = np.zeros((1, hidden_size))

        self.W_cx = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.W_ch = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.b_c = np.zeros((1, hidden_size))

        self.h_list = []  # 保存各时间步隐藏状态
        self.c_list = []  # 保存各时间步记忆细胞状态

    def forward(self, X):
        """
        X形状：[seq_len, batch_size, input_size]
        """
        self.X = X  # 保存输入序列供反向传播使用
        seq_len, batch_size, _ = X.shape
        self.h_list = [np.zeros((self.hidden_size, batch_size))]  # 初始隐藏状态
        self.c_list = [np.zeros((self.hidden_size, batch_size))]  # 初始记忆细胞
        # 新增：保存门控值的列表
        self.i_list = []  # 输入门
        self.f_list = []  # 遗忘门
        self.o_list = []  # 输出门
        self.c_tilde_list = []  # 候选记忆细胞

        for t in range(seq_len):
            x_t = X[t]  # [batch_size, input_size]
            h_prev = self.h_list[-1]  # [hidden_size, batch_size]
            c_prev = self.c_list[-1]  # [hidden_size, batch_size]

            # 计算输入门 i_t
            i_t = self.sigmoid(
                np.matmul(x_t, self.W_ix) + np.matmul(h_prev.T, self.W_ih) + self.b_i)  # [batch_size, hidden_size]
            self.i_list.append(i_t)  # 保存输入门

            # 计算遗忘门 f_t
            f_t = self.sigmoid(
                np.matmul(x_t, self.W_fx) + np.matmul(h_prev.T, self.W_fh) + self.b_f)  # [batch_size, hidden_size]
            self.f_list.append(f_t)  # 保存遗忘门

            # 计算输出门 o_t
            o_t = self.sigmoid(
                np.matmul(x_t, self.W_ox) + np.matmul(h_prev.T, self.W_oh) + self.b_o)  # [batch_size, hidden_size]
            self.o_list.append(o_t)  # 保存输出门

            # 计算候选记忆细胞 c_tilde_t
            c_tilde_t = self.activation_c(
                np.matmul(x_t, self.W_cx) + np.matmul(h_prev.T, self.W_ch) + self.b_c)  # [batch_size, hidden_size]
            self.c_tilde_list.append(c_tilde_t)  # 保存候选记忆细胞

            # 更新记忆细胞 c_t（关键修复：补充c_t的计算）
            c_t = f_t * c_prev.T + i_t * c_tilde_t  # [batch_size, hidden_size]
            c_t = c_t.T  # 转换为 [hidden_size, batch_size]

            # 更新隐藏状态 h_t（使用正确计算的c_t）
            h_t = o_t * self.activation_h(c_t.T)  # [batch_size, hidden_size]
            h_t = h_t.T  # 转换为 [hidden_size, batch_size]

            self.h_list.append(h_t)
            self.c_list.append(c_t)

        # 返回所有时间步的隐藏状态（形状：[seq_len, batch_size, hidden_size]）
        return np.array(self.h_list[1:]).transpose(0, 2, 1)  # 关键修改：调整转置顺序为 (0, 2, 1)


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def backward(self, dh_output):
        """
        LSTM反向传播（BPTT）
        :param dh_output: 输出层误差梯度（形状：[seq_len, batch_size, hidden_size]）
        :return: 参数梯度字典（包含dW_ix, dW_ih, db_i等）
        """
        seq_len, batch_size, _ = dh_output.shape
        dh_output = dh_output.transpose(0, 2, 1)  # 转换为[seq_len, hidden_size, batch_size]
        # print(f"[调试] dh_output转置后形状：{dh_output.shape}（应为[seq_len, hidden_size, batch_size]）")
        
        # 初始化梯度
        dW_ix = np.zeros_like(self.W_ix)
        dW_ih = np.zeros_like(self.W_ih)
        db_i = np.zeros_like(self.b_i)
        
        dW_fx = np.zeros_like(self.W_fx)
        dW_fh = np.zeros_like(self.W_fh)
        db_f = np.zeros_like(self.b_f)
        
        dW_cx = np.zeros_like(self.W_cx)
        dW_ch = np.zeros_like(self.W_ch)
        db_c = np.zeros_like(self.b_c)
        
        dW_ox = np.zeros_like(self.W_ox)
        dW_oh = np.zeros_like(self.W_oh)
        db_o = np.zeros_like(self.b_o)
        
        # 初始梯度（来自后续时间步，初始为0）
        dh_next = np.zeros((self.hidden_size, batch_size))
        dc_next = np.zeros((self.hidden_size, batch_size))
        
        # 从最后一个时间步向前传播
        for t in reversed(range(seq_len)):
            # 当前时间步的隐藏状态梯度（dh_t = dh_output[t] + dh_next）
            dh_t = dh_output[t] + dh_next  # [hidden_size, batch_size]
            # print(f"\n[调试] 时间步t={t}")
            # print(f"[调试] dh_t形状：{dh_t.shape}（应为[hidden_size, batch_size]）")
            
            # 当前时间步的隐藏状态和记忆细胞（已转置）
            h_t = self.h_list[t+1]  # h_1~h_T
            c_t = self.c_list[t+1]  # c_1~c_T
            c_prev = self.c_list[t]  # c_0~c_{T-1}
            o_t = self.o_list[t]
            c_tilde_t = self.c_tilde_list[t]
            i_t = self.i_list[t]
            f_t = self.f_list[t]
            # print(f"[调试] i_t形状：{i_t.shape}（应为[batch_size, hidden_size]）")
            # print(f"[调试] f_t形状：{f_t.shape}（应为[batch_size, hidden_size]）")
            # print(f"[调试] c_tilde_t形状：{c_tilde_t.shape}（应为[batch_size, hidden_size]）")
            
            # 计算记忆细胞梯度（修正o_t的维度）
            dc_t = dh_t * o_t.T * self.activation_h_deriv(c_t.T).T + dc_next  # o_t转置后形状匹配
            # print(f"[调试] dc_t形状：{dc_t.shape}（应为[hidden_size, batch_size]）")

            # 计算输出门梯度（修正维度）
            # 原错误：o_t.T导致维度翻转，改为直接使用o_t的原始维度
            do_t = dh_t.T * self.activation_h(c_t.T) * o_t * (1 - o_t)  # do_t形状：[batch_size, hidden_size]
            # print(f"[调试] 修正后do_t形状：{do_t.shape}（应为[batch_size, hidden_size]）")
            # 获取当前时间步输入x_t（转置为[input_size, batch_size]）
            x_t = self.X[t].T  # 提前到使用前定义
            # print(f"[调试] x_t.T形状：{x_t.shape}（应为[input_size, batch_size]）")
            # 输出门梯度（修正矩阵乘法顺序）
            dW_ox += np.matmul(x_t, do_t)  # 此时x_t已定义
            dW_oh += np.matmul(self.h_list[t], do_t)  # h_prev.T [batch_size, hidden_size] × do_t [batch_size, hidden_size] → [hidden_size, hidden_size]
            db_o += np.sum(do_t, axis=0, keepdims=True)  # 偏置梯度按batch维度求和（形状[1, hidden_size]）
            
            # 计算候选记忆细胞梯度（修正导数输入和转置）
            dc_tilde_t = dc_t * i_t.T * self.activation_c_deriv(c_tilde_t).T  # 关键修改：使用c_tilde_t原始值求导并转置
            # print(f"[调试] dc_tilde_t形状：{dc_tilde_t.shape}（应为[hidden_size, batch_size]）")
            
            # 计算输入门梯度（维度已对齐）
            di_t = dc_t.T * c_tilde_t * i_t * (1 - i_t)  # dc_t.T形状：[batch_size, hidden_size]
            # print(f"[调试] di_t形状：{di_t.shape}（应为[batch_size, hidden_size]）")
            
            # 计算遗忘门梯度（修正维度）
            df_t = dc_t.T * c_prev.T * f_t * (1 - f_t)  # dc_t.T和c_prev.T形状：[batch_size, hidden_size]
            # print(f"[调试] df_t形状：{df_t.shape}（应为[batch_size, hidden_size]）")
            

            
            # 计算输出门梯度（修正维度）
            do_t = dh_t.T * self.activation_h(c_t.T) * o_t * (1 - o_t)  # [batch_size, hidden_size]
            # print(f"[调试] 修正后do_t形状：{do_t.shape}（应为[batch_size, hidden_size]）")
            
            # 输出门梯度（修正矩阵乘法顺序）
            dW_ox += np.matmul(x_t, do_t)  # x_t [input_size, batch_size] × do_t [batch_size, hidden_size] → [input_size, hidden_size]
            dW_oh += np.matmul(self.h_list[t], do_t)  # h_prev [hidden_size, batch_size] × do_t [batch_size, hidden_size] → [hidden_size, hidden_size]
            db_o += np.sum(do_t, axis=0, keepdims=True)  # 偏置梯度按hidden_size维度求和后转置（形状[1, hidden_size]）
            
            # 计算传递到前一时间步的梯度
            dh_next = (
                np.matmul(di_t, self.W_ih) +  # 输入门对h_prev的梯度
                np.matmul(df_t, self.W_fh) +  # 遗忘门对h_prev的梯度
                np.matmul(dc_tilde_t.T, self.W_ch) +  # 候选记忆细胞对h_prev的梯度
                np.matmul(do_t, self.W_oh)  # 输出门对h_prev的梯度
            ).T  # 转置回[hidden_size, batch_size]
            
            # 关键修改：转置f_t以匹配dc_t的维度
            dc_next = dc_t * f_t.T  # dc_t形状：[hidden_size, batch_size]，f_t.T形状：[hidden_size, batch_size]
        
        # 梯度归一化（按批量大小平均）
        dW_ix /= batch_size
        dW_ih /= batch_size
        db_i /= batch_size
        dW_fx /= batch_size
        dW_fh /= batch_size
        db_f /= batch_size
        dW_cx /= batch_size
        dW_ch /= batch_size
        db_c /= batch_size
        dW_ox /= batch_size
        dW_oh /= batch_size
        db_o /= batch_size
        
        # print(f"\n[调试] 反向传播完成，总时间步：{seq_len}，批量大小：{batch_size}")
        # print(f"[调试] 输入门权重梯度dW_ix形状：{dW_ix.shape}（应为[input_size, hidden_size]）")
        # print(f"[调试] 遗忘门权重梯度dW_fx形状：{dW_fx.shape}（应为[input_size, hidden_size]）")
        return {
            "dW_ix": dW_ix, "dW_ih": dW_ih, "db_i": db_i,
            "dW_fx": dW_fx, "dW_fh": dW_fh, "db_f": db_f,
            "dW_cx": dW_cx, "dW_ch": dW_ch, "db_c": db_c,
            "dW_ox": dW_ox, "dW_oh": dW_oh, "db_o": db_o
        }

# ==================== 调试验证 ====================
if __name__ == "__main__":
    # 测试参数
    input_size = 10  # 输入特征维度
    hidden_size = 20  # 隐藏状态维度
    seq_len = 5      # 序列长度
    batch_size = 3   # 批量大小
    
    # 初始化LSTM层
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
    
    # 生成随机输入（形状：[seq_len, batch_size, input_size]）
    X = np.random.randn(seq_len, batch_size, input_size)
    
    # 前向传播
    h_output = lstm.forward(X)
    # print(f"前向传播输出形状：{h_output.shape}（应为[5, 3, 20]）")
    
    # 生成随机输出误差梯度（模拟下游任务的梯度）
    dh_output = np.random.randn(seq_len, batch_size, hidden_size)
    
    # 反向传播
    grads = lstm.backward(dh_output)
    # print(f"输入门偏置梯度db_i形状：{grads['db_i'].shape}（应为[20, 1]）")