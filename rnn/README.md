## Rnn相关整理

本质代码基本上是复刻 Dive-into-DL-Pytorch.pdf

虽说是复刻，但是在理代码的时候发现了一些有趣的

1. seq2seq时，要注意如何处理曝光误差
2. 本次主要实现的是单向rnn，双向的在后面
3. rnn的梯度爆炸和梯度消失，除了pdf中提到裁剪，应该还有用relu作激活函数
4. 时序数据采样：随机采样和相邻采样
5. 困惑度用来评价交叉熵相关的训练指标，困惑度是对交叉熵损失函数做指数运算后得到的值
6. 有些值得注意的pytorch中的精巧用法
   * scatter_ 实现onehot
   * CrossEntropyLoss，包含了softmax + cross，而且y标签不需要onehot化
   * contiguous与view结合使用