import numpy as np

class ActivationFunctions:
    """
    激活函数库，包含tanh、LeakyReLU、ELU及其导数实现
    支持通过字符串参数选择激活函数（如 "tanh"、"leaky_relu"、"elu"）
    """
    
    @staticmethod
    def tanh(x):
        """
        tanh激活函数
        :param x: 输入数组（shape任意）
        :return: tanh(x)
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """
        tanh导数（输入为原始x或直接使用输出值优化计算）
        公式：d(tanh(x))/dx = 1 - tanh(x)^2
        :param x: 输入数组（与tanh的输入x相同）
        :return: 导数数组（shape与x一致）
        """
        tanh_x = ActivationFunctions.tanh(x)
        return 1 - np.square(tanh_x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        LeakyReLU激活函数
        :param x: 输入数组（shape任意）
        :param alpha: 负区间斜率（默认0.01）
        :return: LeakyReLU(x)
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """
        LeakyReLU导数
        公式：x>0时导数为1，x<=0时导数为alpha
        :param x: 输入数组（与LeakyReLU的输入x相同）
        :param alpha: 负区间斜率（默认0.01）
        :return: 导数数组（shape与x一致）
        """
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x, alpha=1.0):
        """
        ELU激活函数
        :param x: 输入数组（shape任意）
        :param alpha: 负区间系数（默认1.0）
        :return: ELU(x)
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))  # 等价于 alpha*(exp(x)-1) 当x<=0
    
    @staticmethod
    def elu_derivative(x, alpha=1.0):
        """
        ELU导数
        公式：x>0时导数为1，x<=0时导数为alpha*exp(x)
        :param x: 输入数组（与ELU的输入x相同）
        :param alpha: 负区间系数（默认1.0）
        :return: 导数数组（shape与x一致）
        """
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def get_activation(activation_name):
        """
        根据名称获取激活函数及其导数
        :param activation_name: 激活函数名称（"tanh"/"leaky_relu"/"elu"）
        :return: (激活函数, 导数函数) 元组
        """
        activation_map = {
            "tanh": (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            "leaky_relu": (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative),
            "elu": (ActivationFunctions.elu, ActivationFunctions.elu_derivative)
        }
        if activation_name not in activation_map:
            raise ValueError(f"不支持的激活函数：{activation_name}，可选：{list(activation_map.keys())}")
        return activation_map[activation_name]

# ==================== 调试验证 ====================
if __name__ == "__main__":
    # 测试输入（包含正负值）
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # 测试tanh
    tanh_out = ActivationFunctions.tanh(x)
    tanh_grad = ActivationFunctions.tanh_derivative(x)
    print(f"tanh({x}) = {tanh_out}")
    print(f"tanh导数: {tanh_grad}\n")  # 应满足 1 - tanh(x)^2
    
    # 测试LeakyReLU（alpha=0.1）
    lr_out = ActivationFunctions.leaky_relu(x, alpha=0.1)
    lr_grad = ActivationFunctions.leaky_relu_derivative(x, alpha=0.1)
    print(f"LeakyReLU({x}, alpha=0.1) = {lr_out}")
    print(f"LeakyReLU导数: {lr_grad}\n")  # 负值位置导数应为0.1
    
    # 测试ELU（alpha=1.0）
    elu_out = ActivationFunctions.elu(x)
    elu_grad = ActivationFunctions.elu_derivative(x)
    print(f"ELU({x}) = {elu_out}")
    print(f"ELU导数: {elu_grad}\n")  # 负值位置导数应为exp(x)（当alpha=1时）