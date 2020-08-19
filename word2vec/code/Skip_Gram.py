import torch
from torch import nn
from .Dataset import get_data_iter_negative_sampling, get_data_iter_h_softmax
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class SkipGram:
    def __init__(self, method, embeddings_size=100, batch_size=512):
        # 这里用sequential更容易将parameters合在一起处理
        if method == 'negative':
            self.data_iter, vocabulary_size = get_data_iter_negative_sampling(batch_size=512)
            self.embeddings_size = embeddings_size
            self.net = nn.Sequential(
                nn.Embedding(num_embeddings=vocabulary_size,
                             embedding_dim=embeddings_size),
                nn.Embedding(num_embeddings=vocabulary_size,
                             embedding_dim=embeddings_size))
            self.net.to(device)

        else:
            self.data_iter, self.n, vocabulary_size = get_data_iter_h_softmax()
            self.embeddings_size = embeddings_size
            self.net = nn.Sequential(
                nn.Embedding(num_embeddings=vocabulary_size,
                             embedding_dim=embeddings_size),
                nn.Embedding(num_embeddings=self.n,
                             embedding_dim=embeddings_size))
            self.net.to(device)

    def train_negative_sampling(self, lr, num_epochs):
        print('train on', device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            l_sum, n = 0.0, 0
            for batch in self.data_iter:
                center, context_negative, mask, label = [d.to(device) for d in batch]
                # permute将tensor的维度换位
                pred = torch.bmm(self.net[0](center), self.net[1](context_negative).permute(0, 2, 1))

                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                # 一个把batch的平均loss
                loss = SigmoidBinaryCrossEntropyLoss()
                l = (loss(pred.view(label.shape), label, mask) *
                     mask.shape[1] / mask.float().sum(dim=1)).mean()

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1
            print('epoch %d, loss %.2f'% (epoch + 1, l_sum / n))

    def train_h_softmax(self, lr, num_epochs):
        print('train on', device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            l_sum, n = 0.0, 0
            for batch in self.data_iter:
                center, signals, parents = [d.to(device) for d in batch]
                new_signals = signals.unsqueeze(-1)
                temp = (self.net[1](parents) * new_signals).permute(0, 2, 1)
                mul = torch.bmm(self.net[0](center), temp)
                mul = torch.sigmoid(mul)
                temp = -torch.log(mul)
                temp = temp * signals * signals
                l = temp.mean()

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1

            print('epoch %d, loss %.2f' % (epoch + 1, l_sum / n))


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        :param inputs: Tensor shape: (batch_size, len)
        :param targets:
        :param mask:
        :return:
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets,
                                                             reduction='none', weight=mask)
        return res.mean(dim=1)




