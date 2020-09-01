# 准备工作
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


import sys
sys.path.append('../..')
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) =  d2l.load_data_jay_lyrics()

# one-hot，后续也可以处理成词向量
def one_hot(x, n_class, dtype=torch.float32):
    # x shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1) # 按照dim=1，也就是列进行填充，参考解释https://www.cnblogs.com/dogecheng/p/11938009.html
    return res

def to_onehot(x, n_class):
    return [one_hot(x[:, i], n_class) for i in range(x.shape[1])]

# 初始化模型参数
# 感觉这个初始化方式很奇妙
num_inputs, num_hiddens, num_outputs = vocab_size, 256,  vocab_size
print('will use', device)

def get_params():
    def _ones(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape),
                          device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    # w是网络权重， b是偏置项
    # rnn 除了有输入，上一层的隐藏层也要输入
    # b不用随机初始化，因为w随机之后，已经打破对称，b就一个常数，无所谓了
    w_xh = _ones((num_inputs, num_hiddens))
    w_hh = _ones((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens,
                                         device=device, requires_grad=True))

    # 输出层参数
    w_hq = _ones((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    return nn.ParameterList([w_xh, w_hh, b_h, w_hq, b_q])

# 定义模型
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# 定义了在⼀个时间步⾥如何计算隐藏状态和输出
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    w_xh, w_hh, b_h, w_hq, b_q = params

    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, w_xh) + torch.matmul(H,w_hh) + b_h)
        # y 这一层没有激活函数
        Y = torch.matmul(H, w_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上⼀时间步的输出作为当前时间步的输⼊
        x = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(x, state, params)
        # 下⼀个时间步的输⼊是prefix⾥的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

# 裁剪梯度
# 循环神经⽹络中较容易出现梯度衰减或梯度爆炸
# 为了应对梯度爆炸，我们可以裁剪梯度
# 假设我们把所有模型参数梯度的元素拼接成一个向量g，设置裁剪的阈值是 theta
# 裁剪后的梯度的L2范数阈值不会超过 theta

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

'''
1. 使⽤困惑度评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采⽤不同采样⽅法将导致隐藏状态初始化的不同。相
'''
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device,
                          corpus_indices, idx_to_char, char_to_idx, is_random_iter, num_epochs,
                          num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    # 6.3节部分讲述的
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()
    state = None
    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使⽤相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size,
                                 num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使⽤随机采样，在每个⼩批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # 否则需要使⽤detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖⼀次迭代读取的⼩批量序列(防⽌梯度计算开销太⼤)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成⻓度为
            # batch * num_steps 的向量，这样跟输出的⾏⼀⼀对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使⽤交叉熵损失计算平均分类误差
            l = loss(outputs, y.long()) # https://www.jianshu.com/p/6049dbc1b73f y没有转换成onehot也可以计算

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()  # optimizer.zero_grad()
            l.backward() # 反向传播计算
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不⽤再做平均, 相当于optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() -
                start)) # 困惑度是对交叉熵损失函数做指数运算后得到的值


            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens,
                                        vocab_size, device, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)