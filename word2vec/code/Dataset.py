import torch
import sys
import torch.utils.data as Data
from .DataProcess import DataProcess


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives=None, signals=None, word_parents=None, cbow=False):
        assert len(centers) == len(contexts)

        if negatives != None:
            assert len(centers) == len(negatives)

        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
        self.word_parents = word_parents
        self.signals = signals
        self.cbow = cbow

    def __getitem__(self, index):
        if self.negatives != None:
            return [self.centers[index], self.contexts[index],
                    self.negatives[index]]
        else:
            if not self.cbow:
                contexts = self.contexts[index]
                signals = []
                parents = []
                for context in contexts:
                    signals.append(self.signals[context])
                    parents.append(self.word_parents[context])

                return [self.centers[index], signals, parents]
            else:

                center = self.centers[index]
                signals = []
                parents = []
                signals.append(self.signals[center])
                parents.append(self.word_parents[center])

                return [self.contexts[index], signals, parents]

    def __len__(self):
        return len(self.centers)

# 在构造⼩批量时，我们将每个样本的背景词和噪声词连结在⼀起，
# 并添加填充项0直⾄连结后的⻓度相同，即⻓度均为 （ max_len 变量）


def batchify_negative(data):
    """用作DataLoader的参数collate_fn：输入是个长为batchsize的list，
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len =  max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []

    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives  += [context + negative + [0] *(max_len-cur_len)]
        # 用下掩码mask，记录那些是填充项
        masks += [[1]*cur_len + [0]*(max_len-cur_len)]
        labels += [[1]*len(context) + [0]*(max_len-len(context))]
    return (torch.tensor(centers).view(-1, 1),
            torch.tensor(contexts_negatives),
            torch.tensor(masks),
            torch.tensor(labels))


def batchify_h_softmax(data):
    centers, signals, parents = [], [], []
    max_len_context = 0
    max_len_parent = 0
    for _,  _, parent in data:
        max_len_context = max(max_len_context, len(parent))
        for p in parent:
            max_len_parent = max(max_len_parent, len(p))

    for center, signal, parent in data:
        cur_len_context = len(parent)
        for i in range(cur_len_context):
            signal[i] += [0] * (max_len_parent - len(signal[i]))
            parent[i] += [0] * (max_len_parent - len(parent[i]))
        parent += [[0] * max_len_parent] * (max_len_context - cur_len_context)
        signal += [[0] * max_len_parent] * (max_len_context - cur_len_context)
        centers += [center]
        signals += signal
        parents += parent

    return (torch.tensor(centers).view(-1, 1),
            torch.tensor(signals).view(-1, max_len_context*max_len_parent),
            torch.tensor(parents).view(-1, max_len_context*max_len_parent))


def batchify_h_softmax_cbow(data):
    contexts, signals, parents, masks = [], [], [], []
    max_len_context = 0
    max_len_parent = 0

    for context,  _, parent in data:
        max_len_context = max(max_len_context, len(context))
        max_len_parent = max(max_len_parent, len(parent[0]))

    for context, signal, parent in data:
        cur_len_context = len(context)
        masks += [[1] * cur_len_context + [0] * (max_len_context - cur_len_context)]
        contexts += [context + [0] *(max_len_context-cur_len_context)]

        signals += [signal[0] + [0] * (max_len_parent - len(signal[0]))]
        parents += [parent[0] + [0] * (max_len_parent - len(parent[0]))]

    return (torch.tensor(contexts),
            torch.tensor(masks),
            torch.tensor(signals),
            torch.tensor(parents))


def get_data_iter_negative_sampling(batch_size=512):
    num_workers = 0 if sys.platform.startswith('win32') else 4
    # 处理数据，获取中心词以及背景词
    dataprocess = DataProcess()
    dataprocess.tokenize(prefix='../../../data/', file='ptb.train.txt')
    dataprocess.sub_sample()
    counter = dataprocess.model_save['counter']
    length = len(counter)
    sampling_weights = [counter[w] ** 0.75 for w in range(length)]
    dataprocess.get_negatives(sampling_weights=sampling_weights, K=5, max_window_size=5)
    dataset = MyDataset(dataprocess.model_save['all_centers'],
                        dataprocess.model_save['all_contexts'],
                        dataprocess.model_save['all_negatives'])
    # 准备迭代用的数据
    data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=batchify_negative,
                                num_workers=num_workers,
                                drop_last=True)
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'], batch):
            print(name, 'shape:', data.shape)
        break

    return data_iter, dataprocess.model_save['vocabulary_size']


def get_data_iter_h_softmax(batch_size=512, method='skipgram'):
    num_workers = 0 if sys.platform.startswith('win32') else 4
    # 处理数据，获取中心词以及背景词
    dataprocess = DataProcess()
    dataprocess.tokenize(prefix='../../../data/', file='ptb.train.txt')
    dataprocess.sub_sample()
    centers, contexts = dataprocess.get_centers_and_contexts(max_window_size=5)
    dataprocess.huffle_tree()
    n, word_parents, signals = dataprocess.model_save['huffle_tree']
    data_iter = None
    if method == 'skipgram':
        dataset = MyDataset(centers,
                            contexts,
                            signals=signals,
                            word_parents=word_parents)
        data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                                    collate_fn=batchify_h_softmax,
                                    num_workers=num_workers)
        for batch in data_iter:
            for name, data in zip(['centers', 'signals', 'parents'], batch):
                print(name, 'shape:', data.shape)
            break
    else:
        dataset = MyDataset(centers,
                            contexts,
                            signals=signals,
                            word_parents=word_parents,
                            cbow=True)

        data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                                    collate_fn=batchify_h_softmax_cbow,
                                    num_workers=num_workers)
        for batch in data_iter:
            for name, data in zip(['contexts', 'masks', 'signals', 'parents'], batch):
                print(name, 'shape:', data.shape)
            break
    return data_iter, n, dataprocess.model_save['vocabulary_size']
